"""
Hydrogen line manual testing and background scanning with web dashboard for remote access
"""
from src.hydrogen_scanner import HydrogenScanner
import time


def main():
    """Main execution function"""
    print("Web-Controlled Hydrogen Line Scanner v4.0")
    print("=" * 60)
    print("All controls available through web interface")
    print("Navigate to http://localhost:5000 to control the scanner")
    print("=" * 60)

    scanner = HydrogenScanner()

    try:
        # Start web server
        scanner.start_web_server()

        # Keep server running
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
