import asyncio
import os
import pickle
import argparse
import struct

SOCK = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'unix_sock')

async def read_pickled(reader):
    sz = await reader.readexactly(4)
    sz = struct.unpack('!L', sz)[0]
    raw = await reader.readexactly(sz)
    return pickle.loads(raw)

def write_pickled(writer, data):
    encoded = pickle.dumps(data, protocol=-1)
    writer.write(struct.pack('!L', len(encoded)))
    writer.write(encoded)

async def handle_connection(reader, writer):
    msg = await read_pickled(reader)
    print('server received:', msg)

    write_pickled(writer, msg)

    await writer.drain()
    writer.close()
    await writer.wait_closed()

async def server_main():
    server = await asyncio.start_unix_server(handle_connection, SOCK)
    async with server:
        await server.serve_forever()

def server():
    try:
        asyncio.run(server_main())
    except KeyboardInterrupt:
        print()
    finally:
        os.remove(SOCK)

async def client_main():
    reader, writer = await asyncio.open_unix_connection(SOCK)

    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--week-view', nargs='?', type=int, default=0)
    args = parser.parse_args(['-w7'])

    msg = ('week', args)
    print('client sending:', msg)
    write_pickled(writer, msg)

    rcv = await read_pickled(reader)
    print('client received:', rcv)

    await writer.drain()
    writer.close()
    await writer.wait_closed()

def client():
    asyncio.run(client_main())
