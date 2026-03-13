"""
Argus System Tray Application
=============================

Runs Argus in the background with a system tray icon.
Supports:
- Minimize to tray
- Status notifications
- Quick access to logs and controls
"""

import asyncio
import sys
import threading
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pystray
    from PIL import Image
except ImportError:
    print("Required packages not installed. Run:")
    print("  pip install pystray pillow")
    sys.exit(1)

from src.orchestrator import ArgusOrchestrator


class ArgusTray:
    """System tray application for Argus."""
    
    def __init__(self):
        self.icon = None
        self.orchestrator = None
        self._running = False
        self._loop = None
        self._thread = None
        
        # Load icon
        icon_path = Path(__file__).parent.parent / "assets" / "argus_icon.png"
        if icon_path.exists():
            self.image = Image.open(icon_path)
        else:
            # Fallback: create simple icon
            self.image = Image.new('RGB', (64, 64), color=(40, 44, 52))
    
    def create_menu(self):
        """Create system tray menu."""
        return pystray.Menu(
            pystray.MenuItem("Argus Market Monitor", None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                "Status: Running" if self._running else "Status: Stopped",
                None,
                enabled=False
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("View Logs", self.open_logs),
            pystray.MenuItem("Check Database", self.check_database),
            pystray.MenuItem("Open Config Folder", self.open_config),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Stop Argus", self.stop_argus),
            pystray.MenuItem("Quit", self.quit_app),
        )
    
    def open_logs(self, icon=None, item=None):
        """Open logs folder."""
        logs_path = Path(__file__).parent.parent / "logs"
        if logs_path.exists():
            os.startfile(str(logs_path))
        else:
            logs_path.mkdir(exist_ok=True)
            os.startfile(str(logs_path))
    
    def check_database(self, icon=None, item=None):
        """Run database check script."""
        script_path = Path(__file__).parent / "check_db.py"
        os.system(f'start cmd /k python "{script_path}"')
    
    def open_config(self, icon=None, item=None):
        """Open config folder."""
        config_path = Path(__file__).parent.parent / "config"
        os.startfile(str(config_path))
    
    def stop_argus(self, icon=None, item=None):
        """Stop Argus gracefully."""
        if self._running and self.orchestrator:
            self._running = False
            if self._loop:
                asyncio.run_coroutine_threadsafe(
                    self.orchestrator.stop(),
                    self._loop
                )
            self.icon.notify("Argus Stopped", "Market monitoring has been stopped")
    
    def quit_app(self, icon=None, item=None):
        """Quit the application."""
        self.stop_argus()
        self.icon.stop()
    
    async def run_orchestrator(self):
        """Run the Argus orchestrator."""
        self._loop = asyncio.get_event_loop()
        self.orchestrator = ArgusOrchestrator()
        self._running = True
        
        try:
            await self.orchestrator.setup()
            self.icon.notify("Argus Started", "Market monitoring is now active")
            await self.orchestrator.run()
        except Exception as e:
            self.icon.notify("Argus Error", str(e))
        finally:
            self._running = False
    
    def start_orchestrator_thread(self):
        """Start Argus in a background thread."""
        def run():
            asyncio.run(self.run_orchestrator())
        
        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
    
    def run(self):
        """Run the system tray application."""
        # Single instance check
        lock_file = Path(os.getenv('TEMP')) / "argus.lock"
        
        try:
            # Try to remove stale lock file if process isn't actually running
            if lock_file.exists():
                try:
                    os.remove(lock_file)
                except OSError:
                    # File is locked, meaning instance IS running
                    print("Argus is already running!")
                    return

            # Create lock file
            with open(lock_file, 'w') as f:
                f.write(str(os.getpid()))
            
            # Use file locking to ensure it stays locked while running
            # (Windows-specific dumb simple approach: keep file handle open)
            self._lock_handle = open(lock_file, 'r')

            self.icon = pystray.Icon(
                "Argus",
                self.image,
                "Argus Market Monitor",
                menu=self.create_menu()
            )
            
            # Start Argus in background
            self.start_orchestrator_thread()
            
            # Run tray icon (blocks)
            self.icon.run()
            
        finally:
            # Cleanup lock on exit
            if hasattr(self, '_lock_handle'):
                self._lock_handle.close()
            try:
                if lock_file.exists():
                    os.remove(lock_file)
            except OSError:
                pass


def main():
    """Entry point."""
    # Change to project directory
    os.chdir(Path(__file__).parent.parent)
    
    # Check for existing instance first
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(('localhost', 64123))  # Bind to a specific port
    except socket.error:
        print("Argus is already active.")
        return

    print("Starting Argus in system tray mode...")
    
    tray = ArgusTray()
    
    # Keep socket open to hold the lock
    tray._socket_lock = s
    
    tray.run()


if __name__ == "__main__":
    main()
