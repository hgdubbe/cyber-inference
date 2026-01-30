"""
Automatic whisper.cpp installation and updates.

Handles:
- Platform detection (macOS, Linux, Windows)
- GPU support detection (CUDA, Metal, CPU)
- Binary download from GitHub releases
- Version management and updates
- System binary detection (uses PATH binaries if available)
"""

import os
import platform
import shutil
import stat
import subprocess
import tarfile
import zipfile
from pathlib import Path
from typing import Optional

import httpx

from cyber_inference.core.config import get_settings
from cyber_inference.core.logging import get_logger

logger = get_logger(__name__)

# GitHub releases API for whisper.cpp
WHISPER_CPP_REPO = "ggerganov/whisper.cpp"
GITHUB_API_URL = f"https://api.github.com/repos/{WHISPER_CPP_REPO}/releases/latest"


class WhisperInstaller:
    """
    Manages whisper.cpp installation and updates.

    Automatically detects platform and GPU support to download
    the appropriate binary. Checks system PATH first before
    downloading a new binary.
    """

    def __init__(self, bin_dir: Optional[Path] = None):
        """
        Initialize the installer.

        Args:
            bin_dir: Directory for binaries (default from settings)
        """
        settings = get_settings()
        self.bin_dir = bin_dir or settings.bin_dir
        self.bin_dir.mkdir(parents=True, exist_ok=True)

        self._platform = platform.system().lower()
        self._arch = platform.machine().lower()
        self._gpu_backend: Optional[str] = None

        logger.info("[info]WhisperInstaller initialized[/info]")
        logger.debug(f"  Platform: {self._platform}")
        logger.debug(f"  Architecture: {self._arch}")
        logger.debug(f"  Binary directory: {self.bin_dir}")

    async def detect_gpu_backend(self) -> str:
        """
        Detect the best GPU backend for this system.

        Returns:
            Backend name: 'cuda', 'metal', 'vulkan', or 'cpu'
        """
        logger.info("[info]Detecting GPU backend for whisper.cpp...[/info]")

        # Check for NVIDIA CUDA
        nvidia_smi = shutil.which("nvidia-smi")
        if not nvidia_smi:
            fallback = Path("/usr/bin/nvidia-smi")
            if fallback.exists():
                nvidia_smi = str(fallback)

        if nvidia_smi:
            try:
                result = subprocess.run(
                    [nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0 and result.stdout.strip():
                    logger.info("[success]Detected NVIDIA GPU - using CUDA backend[/success]")
                    self._gpu_backend = "cuda"
                    return "cuda"
            except Exception as e:
                logger.debug(f"CUDA detection failed: {e}")
        elif Path("/proc/driver/nvidia/gpus").exists():
            logger.info("[success]Detected NVIDIA driver via /proc - using CUDA backend[/success]")
            self._gpu_backend = "cuda"
            return "cuda"

        # Check for Apple Metal (macOS)
        if self._platform == "darwin":
            logger.info("[success]macOS detected - using Metal backend[/success]")
            self._gpu_backend = "metal"
            return "metal"

        # Check for Vulkan
        if shutil.which("vulkaninfo"):
            try:
                result = subprocess.run(
                    ["vulkaninfo", "--summary"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    logger.info("[success]Vulkan support detected[/success]")
                    self._gpu_backend = "vulkan"
                    return "vulkan"
            except Exception as e:
                logger.debug(f"Vulkan detection failed: {e}")

        logger.info("[warning]No GPU acceleration detected - using CPU backend[/warning]")
        self._gpu_backend = "cpu"
        return "cpu"

    def _get_release_asset_name(self, backend: str) -> str:
        """
        Get the appropriate release asset name for this platform.

        Args:
            backend: GPU backend (cuda, metal, cpu)

        Returns:
            Asset filename pattern to match
        """
        # whisper.cpp release naming convention
        if self._platform == "darwin":
            # macOS - whisper.cpp uses "macos" or similar
            if self._arch in ("arm64", "aarch64"):
                return r"whisper-.*-bin-macos-arm64"
            else:
                return r"whisper-.*-bin-macos-x64"

        elif self._platform == "linux":
            if backend == "cuda":
                if self._arch in ("arm64", "aarch64"):
                    return r"whisper-.*-bin-linux-arm64-cuda"
                else:
                    return r"whisper-.*-bin-linux-x64-cuda"
            elif self._arch in ("arm64", "aarch64"):
                return r"whisper-.*-bin-linux-arm64"
            else:
                return r"whisper-.*-bin-linux-x64"

        elif self._platform == "windows":
            if backend == "cuda":
                return r"whisper-.*-bin-win-cuda-.*-x64"
            else:
                return r"whisper-.*-bin-win-x64"

        raise ValueError(f"Unsupported platform: {self._platform}")

    async def get_latest_release(self) -> dict:
        """
        Get the latest release info from GitHub.

        Returns:
            Release information dictionary
        """
        logger.info("[info]Fetching latest whisper.cpp release...[/info]")

        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.get(
                GITHUB_API_URL,
                headers={"Accept": "application/vnd.github.v3+json"},
                timeout=30,
            )
            response.raise_for_status()
            release = response.json()

            logger.info(f"[success]Latest release: {release['tag_name']}[/success]")
            logger.debug(f"  Published: {release['published_at']}")
            logger.debug(f"  Assets: {len(release['assets'])}")

            return release

    async def download_file(
        self, url: str, dest: Path, expected_size: Optional[int] = None
    ) -> None:
        """
        Download a file with progress logging.

        Args:
            url: URL to download
            dest: Destination path
            expected_size: Expected file size in bytes
        """
        logger.info(f"[info]Downloading: {url}[/info]")
        logger.debug(f"  Destination: {dest}")

        async with httpx.AsyncClient(follow_redirects=True) as client:
            async with client.stream("GET", url, timeout=300) as response:
                response.raise_for_status()

                total_size = int(response.headers.get("content-length", 0))
                downloaded = 0
                last_log_percent = 0

                with open(dest, "wb") as f:
                    async for chunk in response.aiter_bytes(chunk_size=1024 * 1024):
                        f.write(chunk)
                        downloaded += len(chunk)

                        if total_size > 0:
                            percent = int((downloaded / total_size) * 100)
                            if percent >= last_log_percent + 10:
                                logger.info(f"  Download progress: {percent}%")
                                last_log_percent = percent

        size_mb = downloaded / (1024 * 1024)
        logger.info(f"[success]Download complete: {dest.name} ({size_mb:.1f} MB)[/success]")

    async def extract_archive(self, archive_path: Path, dest_dir: Path) -> None:
        """
        Extract a downloaded archive.

        Args:
            archive_path: Path to the archive
            dest_dir: Destination directory
        """
        logger.info(f"[info]Extracting: {archive_path.name}[/info]")

        if archive_path.suffix == ".zip":
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(dest_dir)
        elif archive_path.suffix in (".gz", ".tgz") or ".tar" in archive_path.name:
            with tarfile.open(archive_path, "r:*") as tf:
                tf.extractall(dest_dir)
        else:
            raise ValueError(f"Unknown archive format: {archive_path.suffix}")

        logger.info("[success]Extraction complete[/success]")

    def _find_whisper_server(self, search_dir: Path) -> Optional[Path]:
        """
        Find the whisper-server binary in the extracted files.

        Args:
            search_dir: Directory to search

        Returns:
            Path to whisper-server binary, or None
        """
        # whisper.cpp server binary names
        patterns = [
            "whisper-server",
            "whisper-server.exe",
            "server",
            "server.exe",
            "main",
            "main.exe",
        ]

        for pattern in patterns:
            for path in search_dir.rglob(pattern):
                if path.is_file() and path.stat().st_mode & stat.S_IXUSR:
                    logger.debug(f"Found whisper-server candidate: {path}")
                    # Verify it's actually whisper-server by checking help output
                    try:
                        result = subprocess.run(
                            [str(path), "--help"],
                            capture_output=True,
                            text=True,
                            timeout=5,
                        )
                        output = result.stdout + result.stderr
                        if "whisper" in output.lower() or "transcri" in output.lower():
                            logger.debug(f"Confirmed whisper-server: {path}")
                            return path
                    except Exception:
                        pass

        # Fallback: look for any executable with 'server' in name
        for path in search_dir.rglob("*server*"):
            if path.is_file():
                logger.debug(f"Found server binary (fallback): {path}")
                return path

        return None

    def _find_system_binary(self) -> Optional[Path]:
        """
        Check if whisper-server exists in system PATH.

        Returns:
            Path to system binary if found, None otherwise
        """
        # Check various possible names for whisper server binary
        binary_names = [
            "whisper-server",
            "whisper-cpp-server",
            "whisper.cpp-server",
        ]

        if self._platform == "windows":
            binary_names = [f"{name}.exe" for name in binary_names]

        for name in binary_names:
            system_path = shutil.which(name)
            if system_path:
                logger.debug(f"Found system whisper binary: {system_path}")
                return Path(system_path)

        return None

    def is_installed(self) -> bool:
        """
        Check if whisper-server is installed.

        Checks system PATH first, then falls back to bin_dir.
        """
        # Check system PATH first
        if self._find_system_binary() is not None:
            return True

        # Fall back to bin_dir
        whisper_server_path = self.bin_dir / "whisper-server"
        if self._platform == "windows":
            whisper_server_path = self.bin_dir / "whisper-server.exe"

        return whisper_server_path.exists()

    def get_binary_path(self) -> Path:
        """
        Get the path to the whisper-server binary.

        Checks system PATH first, then falls back to bin_dir.
        """
        # Check system PATH first
        system_binary = self._find_system_binary()
        if system_binary is not None:
            return system_binary

        # Fall back to bin_dir
        whisper_server_path = self.bin_dir / "whisper-server"
        if self._platform == "windows":
            whisper_server_path = self.bin_dir / "whisper-server.exe"
        return whisper_server_path

    async def install(
        self,
        platform_override: Optional[str] = None,
        force: bool = False,
    ) -> Path:
        """
        Install or update whisper.cpp.

        Args:
            platform_override: Override platform detection
            force: Force reinstall even if already installed

        Returns:
            Path to the whisper-server binary
        """
        logger.info("[highlight]═════════════════════════════════════════════════[/highlight]")
        logger.info("[highlight]           Installing whisper.cpp                [/highlight]")
        logger.info("[highlight]═════════════════════════════════════════════════[/highlight]")

        # Check system PATH first
        system_binary = self._find_system_binary()
        if system_binary and not force:
            logger.info(f"[success]Using system whisper-server: {system_binary}[/success]")
            version = await self.get_installed_version()
            if version:
                logger.info(f"  Version: {version}")
            return system_binary

        # Check if already installed in bin_dir
        whisper_server_path = self.bin_dir / "whisper-server"
        if self._platform == "windows":
            whisper_server_path = self.bin_dir / "whisper-server.exe"

        if whisper_server_path.exists() and not force:
            logger.info(f"[success]whisper-server installed: {whisper_server_path}[/success]")
            version = await self.get_installed_version()
            if version:
                logger.info(f"  Version: {version}")
            return whisper_server_path

        # Detect GPU backend
        backend = await self.detect_gpu_backend()
        logger.info(f"Selected backend: {backend}")

        # Get latest release
        release = await self.get_latest_release()

        # Find matching asset
        import re
        asset_pattern = self._get_release_asset_name(backend)
        logger.debug(f"Looking for asset matching: {asset_pattern}")

        matching_asset = None
        for asset in release["assets"]:
            if re.match(asset_pattern, asset["name"], re.IGNORECASE):
                matching_asset = asset
                break

        if not matching_asset:
            # Fall back to generic build
            logger.warning("[warning]No specific build found, trying generic...[/warning]")
            for asset in release["assets"]:
                name_lower = asset["name"].lower()
                if self._platform in name_lower and "bin" in name_lower:
                    matching_asset = asset
                    break

        if not matching_asset:
            raise RuntimeError(
                f"No compatible whisper.cpp build found for {self._platform} {self._arch}. "
                "You may need to build from source or install whisper-server manually."
            )

        logger.info(f"[info]Selected asset: {matching_asset['name']}[/info]")

        # Download
        download_url = matching_asset["browser_download_url"]
        archive_name = matching_asset["name"]
        archive_path = self.bin_dir / archive_name

        await self.download_file(download_url, archive_path, matching_asset.get("size"))

        # Extract
        extract_dir = self.bin_dir / "whisper_extracted"
        extract_dir.mkdir(exist_ok=True)
        await self.extract_archive(archive_path, extract_dir)

        # Find and copy whisper-server
        server_binary = self._find_whisper_server(extract_dir)
        if not server_binary:
            raise RuntimeError("whisper-server binary not found in archive")

        # Copy to bin directory
        shutil.copy2(server_binary, whisper_server_path)

        # Make executable
        if self._platform != "windows":
            os.chmod(whisper_server_path, os.stat(whisper_server_path).st_mode | stat.S_IEXEC)

        # Copy required dynamic libraries
        source_dir = server_binary.parent
        lib_count = 0

        lib_patterns = []
        if self._platform == "darwin":
            lib_patterns = ["*.dylib"]
        elif self._platform == "linux":
            lib_patterns = ["*.so", "*.so.*"]
        elif self._platform == "windows":
            lib_patterns = ["*.dll"]

        for pattern in lib_patterns:
            for lib_file in source_dir.glob(pattern):
                dest_lib = self.bin_dir / lib_file.name
                if not dest_lib.exists():  # Don't overwrite llama.cpp libs
                    shutil.copy2(lib_file, dest_lib)
                    logger.debug(f"  Copied library: {lib_file.name}")
                    lib_count += 1

        if lib_count > 0:
            logger.info(f"[success]Copied {lib_count} library files[/success]")

        # Cleanup
        archive_path.unlink()
        shutil.rmtree(extract_dir)

        logger.info(f"[success]whisper-server installed: {whisper_server_path}[/success]")

        # Verify
        version = await self.get_installed_version()
        if version:
            logger.info(f"  Version: {version}")

        return whisper_server_path

    async def get_installed_version(self) -> Optional[str]:
        """
        Get the version of the installed whisper-server.

        Returns:
            Version string, or None if not installed
        """
        binary_path = self.get_binary_path()

        if not binary_path.exists():
            return None

        try:
            result = subprocess.run(
                [str(binary_path), "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            version = result.stdout.strip() or result.stderr.strip()
            if version:
                return version

            # Some versions don't have --version, try --help
            result = subprocess.run(
                [str(binary_path), "--help"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            # Extract version from help output if present
            output = result.stdout + result.stderr
            for line in output.split("\n"):
                if "version" in line.lower() or "whisper" in line.lower():
                    return line.strip()[:100]  # Limit length

            return "installed (version unknown)"
        except Exception:
            return "installed (version check failed)"
