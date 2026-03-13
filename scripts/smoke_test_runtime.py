
import asyncio
import platform
import subprocess
from src.agent.runtime_controller import RuntimeController
from src.agent.zeus import ZeusPolicyEngine, RuntimeMode
from src.core.config import ZeusConfig

async def smoke_test():
    print("Starting RuntimeController Windows Smoke Test...")
    
    # 1. Mock Zeus
    config = ZeusConfig(governance_db_path="data/test_zeus.db", ollama_service_name="Ollama")
    zeus = ZeusPolicyEngine(config)
    
    # 2. Mock ResourceManager
    class MockRM:
        gpu_enabled = True
    rm = MockRM()
    
    controller = RuntimeController(zeus, rm)
    
    # 3. Test Transition
    print("Testing transition to DATA_ONLY (Shutdown)...")
    # Start a dummy process that looks like ollama to see if it gets killed
    if platform.system().lower() == "windows":
        # We can't easily start 'ollama.exe' if not installed, but we can test the command execution
        print("Running taskkill check...")
        success = controller._try_windows_process_kill()
        print(f"Taskkill result (success if command ran): {success}")
    else:
        print("Skipping Windows-specific kill test on this OS.")

    print("Smoke test complete.")

if __name__ == "__main__":
    asyncio.run(smoke_test())
