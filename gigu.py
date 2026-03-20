"""
Automated browser session manager with geolocation-aware CDP mode.

Usage:
    python session_manager.py --url "https://example.com" [OPTIONS]

Examples:
    python session_manager.py --url "https://example.com" --brave --incognito
    python session_manager.py --url "https://example.com" --xvfb --screenshot
    python session_manager.py --url "https://example.com" --brave --incognito --xvfb --screenshot --proxy socks5://127.0.0.1:9050
"""

import argparse
import base64
import logging
import os
import platform
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests
from requests.exceptions import RequestException
from seleniumbase import SB

# ─── Logging ────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── Configuration ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GeoData:
    """Immutable geolocation context resolved from the host's public IP."""
    latitude: float
    longitude: float
    timezone_id: str
    country_code: str

    @classmethod
    def from_ip_api(cls, timeout: int = 10) -> "GeoData":
        """Fetch geolocation from ip-api.com with error handling."""
        try:
            response = requests.get(
                "http://ip-api.com/json/",
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()

            if data.get("status") != "success":
                raise ValueError(f"ip-api returned failure: {data.get('message', 'unknown')}")

            return cls(
                latitude=data["lat"],
                longitude=data["lon"],
                timezone_id=data["timezone"],
                country_code=data["countryCode"].lower(),
            )
        except (RequestException, KeyError, ValueError) as exc:
            logger.error("Failed to resolve geolocation: %s", exc)
            raise SystemExit(1) from exc


@dataclass
class SessionConfig:
    """All tunables for a browser session, gathered in one place."""
    target_url: str
    proxy: Optional[str] = None
    locale: str = "en"
    ad_block: bool = True
    disable_webgl: bool = True
    use_brave: bool = False
    incognito: bool = False
    xvfb: bool = False
    screenshot: bool = False
    idle_range: tuple[int, int] = (450, 800)
    max_sessions_per_cycle: int = 2
    consent_button_selector: str = 'button:contains("Accept")'
    start_button_selector: str = 'button:contains("Start Watching")'
    stream_indicator_selector: str = "#live-channel-stream-information"
    page_load_delay: float = 2.0
    stream_load_delay: float = 12.0
    post_action_delay: float = 10.0
    button_click_timeout: float = 4.0

    @property
    def chromium_args(self) -> str:
        """Build the chromium arg string from config flags."""
        args: list[str] = []
        if self.disable_webgl:
            args.append("--disable-webgl")
        if self.incognito:
            args.append("--incognito")
        return ",".join(args) if args else ""

    def validate(self) -> None:
        """Raise on invalid or incompatible flag combinations."""
        if self.xvfb and platform.system() != "Linux":
            raise SystemExit(
                "--xvfb is only supported on Linux (requires Xvfb). "
                f"Current platform: {platform.system()}"
            )

        if self.screenshot and not self.xvfb:
            logger.warning(
                "--screenshot works best with --xvfb on headless servers. "
                "Proceeding anyway, but screenshots may capture a visible desktop."
            )


# ─── Core Logic ─────────────────────────────────────────────────────────────

class BrowserSessionManager:
    """Manages geolocation-aware browser sessions with CDP mode."""

    def __init__(self, config: SessionConfig, geo: GeoData) -> None:
        self._config = config
        self._geo = geo

    # ── Public API ───────────────────────────────────────────────────────

    def run_forever(self) -> None:
        """Main loop: open sessions until the stream goes offline."""
        cycle = 0
        while True:
            cycle += 1
            logger.info("── Cycle %d ──", cycle)

            try:
                stream_live = self._run_cycle()
            except Exception:
                logger.exception("Unhandled error in cycle %d — restarting", cycle)
                continue

            if not stream_live:
                logger.info("Stream appears offline. Exiting.")
                break

    # ── Private helpers ──────────────────────────────────────────────────

    def _build_sb_kwargs(self) -> dict:
        """
        Construct the keyword arguments dict for the SB() context manager.
        Browser choice, display mode, and screenshot options are all
        centralised here so the rest of the code stays agnostic.
        """
        cfg = self._config

        kwargs: dict = {
            "uc": True,
            "locale": cfg.locale,
            "ad_block": cfg.ad_block,
            "proxy": cfg.proxy if cfg.proxy else False,
        }

        # ── Brave ────────────────────────────────────────────────────
        if cfg.use_brave:
            kwargs["browser"] = "chrome"
            kwargs["binary_location"] = _resolve_brave_binary()
            logger.info("Using Brave at: %s", kwargs["binary_location"])

        # ── Chromium args (--disable-webgl, --incognito, …) ──────────
        chromium_args = cfg.chromium_args
        if chromium_args:
            kwargs["chromium_arg"] = chromium_args

        # ── Xvfb (Linux virtual framebuffer) ─────────────────────────
        if cfg.xvfb:
            kwargs["xvfb"] = True
            logger.info("Xvfb virtual display enabled (headless Linux mode)")

        # ── Screenshots ──────────────────────────────────────────────
        if cfg.screenshot:
            kwargs["screenshot_after_test"] = True
            logger.info("Screenshots will be saved after each session")

        return kwargs

    def _run_cycle(self) -> bool:
        """
        Open a primary browser, optionally spawn secondary sessions,
        idle for a random duration, then tear down.

        Returns True if the stream was live, False otherwise.
        """
        cfg = self._config
        sb_kwargs = self._build_sb_kwargs()

        with SB(**sb_kwargs) as driver:
            self._activate_with_geo(driver)
            self._dismiss_consent(driver)

            driver.sleep(cfg.stream_load_delay)
            self._click_if_present(driver, cfg.start_button_selector, post_delay=cfg.post_action_delay)
            self._dismiss_consent(driver)

            if not driver.is_element_present(cfg.stream_indicator_selector):
                logger.warning("Stream indicator not found — stream may be offline.")
                self._take_debug_screenshot(driver, "offline_check")
                return False

            logger.info("Stream is live. Spawning %d extra session(s).", cfg.max_sessions_per_cycle - 1)
            self._dismiss_consent(driver)

            extra_drivers = self._spawn_extra_sessions(driver)

            idle_seconds = random.randint(*cfg.idle_range)
            logger.info("Idling for %d s…", idle_seconds)
            driver.sleep(idle_seconds)

            del extra_drivers

        return True

    def _spawn_extra_sessions(self, primary_driver) -> list:
        """Open additional browser windows reusing the primary driver context."""
        cfg = self._config
        extras = []

        for i in range(1, cfg.max_sessions_per_cycle):
            logger.info("Spawning extra session %d…", i)
            try:
                extra = primary_driver.get_new_driver(undetectable=True)
                self._activate_with_geo(extra)
                extra.sleep(cfg.post_action_delay)

                self._click_if_present(extra, cfg.start_button_selector, post_delay=cfg.post_action_delay)
                self._dismiss_consent(extra)

                extras.append(extra)
            except Exception:
                logger.exception("Failed to spawn extra session %d", i)

        return extras

    def _activate_with_geo(self, driver) -> None:
        """Activate CDP mode with geolocation and timezone spoofing."""
        cfg = self._config
        driver.activate_cdp_mode(
            cfg.target_url,
            tzone=self._geo.timezone_id,
            geoloc=(self._geo.latitude, self._geo.longitude),
        )
        driver.sleep(cfg.page_load_delay)

    def _dismiss_consent(self, driver) -> None:
        """Click away cookie / consent banners if present."""
        self._click_if_present(driver, self._config.consent_button_selector)

    def _click_if_present(
        self,
        driver,
        selector: str,
        post_delay: float = 0.0,
    ) -> bool:
        """Click an element if it exists. Returns True if clicked."""
        if driver.is_element_present(selector):
            try:
                driver.cdp.click(selector, timeout=self._config.button_click_timeout)
                logger.debug("Clicked: %s", selector)
                if post_delay:
                    driver.sleep(post_delay)
                return True
            except Exception:
                logger.warning("Element found but click failed: %s", selector)
        return False

    def _take_debug_screenshot(self, driver, label: str) -> None:
        """Save a timestamped debug screenshot when --screenshot is active."""
        if not self._config.screenshot:
            return
        try:
            filename = f"debug_{label}_{int(time.time())}.png"
            driver.save_screenshot(filename)
            logger.info("Debug screenshot saved: %s", filename)
        except Exception:
            logger.warning("Failed to save debug screenshot: %s", label)


# ─── Utilities ──────────────────────────────────────────────────────────────

def _resolve_brave_binary() -> str:
    """
    Return the Brave binary path for the current platform.
    Raises FileNotFoundError if Brave isn't installed at any known location.
    """
    candidates: list[Path] = []
    system = platform.system()

    if system == "Linux":
        candidates = [
            Path("/usr/bin/brave-browser"),
            Path("/usr/bin/brave"),
            Path("/snap/bin/brave"),
            Path.home() / ".local/bin/brave",
        ]
    elif system == "Darwin":
        candidates = [
            Path("/Applications/Brave Browser.app/Contents/MacOS/Brave Browser"),
        ]
    elif system == "Windows":
        for env_var in ("LOCALAPPDATA", "PROGRAMFILES", "PROGRAMFILES(X86)"):
            base = os.environ.get(env_var)
            if base:
                candidates.append(
                    Path(base) / "BraveSoftware" / "Brave-Browser" / "Application" / "brave.exe"
                )

    for path in candidates:
        if path.exists():
            return str(path)

    raise FileNotFoundError(
        f"Brave binary not found. Searched: {[str(p) for p in candidates]}. "
        "Install Brave or pass a custom path."
    )


def decode_target(encoded: str) -> str:
    """Decode a base64-encoded channel name."""
    try:
        return base64.b64decode(encoded).decode("utf-8")
    except Exception as exc:
        raise ValueError(f"Failed to decode target: {exc}") from exc


# ─── Entry Point ────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Geo-aware browser session manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  %(prog)s --url "https://example.com" --brave --incognito
  %(prog)s --url "https://example.com" --xvfb --screenshot
  %(prog)s --url "https://example.com" --brave --incognito --xvfb --screenshot
        """,
    )

    # ── Required ─────────────────────────────────────────────────────
    parser.add_argument("--url", required=True, help="Target URL to open")

    # ── Network ──────────────────────────────────────────────────────
    parser.add_argument("--proxy", default=None, help="Proxy string (e.g. socks5://host:port)")

    # ── Session tuning ───────────────────────────────────────────────
    parser.add_argument("--max-sessions", type=int, default=2, help="Browser instances per cycle (default: 2)")
    parser.add_argument("--idle-min", type=int, default=450, help="Min idle seconds (default: 450)")
    parser.add_argument("--idle-max", type=int, default=800, help="Max idle seconds (default: 800)")

    # ── Browser selection ────────────────────────────────────────────
    parser.add_argument(
        "--brave",
        action="store_true",
        default=False,
        help="Use Brave browser instead of Chrome",
    )
    parser.add_argument(
        "--incognito",
        action="store_true",
        default=False,
        help="Launch browser in incognito / private mode",
    )

    # ── Linux display & debugging ────────────────────────────────────
    parser.add_argument(
        "--xvfb",
        action="store_true",
        default=False,
        help="Run inside Xvfb virtual framebuffer (Linux only — ideal for headless servers)",
    )
    parser.add_argument(
        "--screenshot",
        action="store_true",
        default=False,
        help="Save screenshots after each session and on errors (pairs well with --xvfb)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("Resolving geolocation…")
    geo = GeoData.from_ip_api()
    logger.info("Geo: %s / %s  TZ=%s", geo.latitude, geo.longitude, geo.timezone_id)

    config = SessionConfig(
        target_url=args.url,
        proxy=args.proxy,
        idle_range=(args.idle_min, args.idle_max),
        max_sessions_per_cycle=args.max_sessions,
        use_brave=args.brave,
        incognito=args.incognito,
        xvfb=args.xvfb,
        screenshot=args.screenshot,
    )

    # ── Validate before starting ─────────────────────────────────────
    config.validate()

    browser_label = "Brave" if config.use_brave else "Chrome"
    mode_label = " (incognito)" if config.incognito else ""
    display_label = " [xvfb]" if config.xvfb else ""
    screenshot_label = " [screenshots on]" if config.screenshot else ""
    logger.info("Browser: %s%s%s%s", browser_label, mode_label, display_label, screenshot_label)

    manager = BrowserSessionManager(config, geo)
    manager.run_forever()


if __name__ == "__main__":
    main()
