from bleak import BleakScanner
import asyncio

async def main():
    print("Scanning 8s...")
    devices = await BleakScanner.discover(timeout=8.0)
    for d in devices:
        print(d.name, ":", d.address)

asyncio.run(main())
