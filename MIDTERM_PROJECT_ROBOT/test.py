import asyncio
from RPLidar import RPLidar
from rplidarc1.rplidarc1.main import RPLidarC1

async def run_lidar():
    # Replace '/dev/tty.your_device_name' with your actual serial port name
    lidar = RPLidarC1(port='/dev/tty.usbserial-1120')
    try:
        await lidar.connect()
        print("Connected to RPLidar C1")
        
        # Check health status
        health = await lidar.get_health()
        print(f"Health status: {health['status']}")

        # Start scanning and process data (example for getting data once)
        # The library provides options for queue-based or dictionary-based data output
        # Refer to the official documentation for detailed scan processing loops
        
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        await lidar.disconnect()

if __name__ == "__main__":
    asyncio.run(run_lidar())

