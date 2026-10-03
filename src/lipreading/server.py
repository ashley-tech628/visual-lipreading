"""Local video demo: the browser decodes video; only sampled frames reach localhost."""
import base64
import io
import json
import threading
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from importlib.resources import files

from PIL import Image
from .inference import predict_frames
from .model import load_bundle

MAX_REQUEST=8*1024*1024


def decode_request(body):
    payload=json.loads(body)
    encoded=payload.get('frames')
    if not isinstance(encoded,list) or not 1<=len(encoded)<=60:raise ValueError("Supply 1 to 60 sampled frames")
    frames=[]
    for item in encoded:
        if not isinstance(item,str) or len(item)>MAX_REQUEST:raise ValueError("Invalid frame encoding")
        binary=base64.b64decode(item,validate=True)
        with Image.open(io.BytesIO(binary)) as image:
            if image.width>2048 or image.height>2048:raise ValueError("Frame exceeds 2048 pixels")
            frames.append(image.convert('RGB').copy())
    roi=payload.get('roi')
    return frames,roi


def make_handler(model,bundle,port):
    lock=threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def reply(self,status,payload,content_type='application/json; charset=utf-8'):
            binary=payload if isinstance(payload,bytes) else json.dumps(payload,ensure_ascii=False).encode('utf8')
            self.send_response(status);self.send_header('Content-Type',content_type)
            self.send_header('Content-Length',str(len(binary)));self.send_header('X-Content-Type-Options','nosniff')
            self.end_headers();self.wfile.write(binary)

        def do_GET(self):
            if self.path=='/':self.reply(200,files('lipreading').joinpath('web/index.html').read_bytes(),'text/html; charset=utf-8')
            elif self.path=='/api/health':self.reply(200,dict(status='ready',labels=bundle['labels'],preprocessing=bundle['preprocessing'],aggregation=bundle['config'].get('aggregation',bundle['config'].get('architecture'))))
            else:self.reply(404,dict(error='Not found'))

        def do_POST(self):
            if self.path!='/api/predict':self.reply(404,dict(error='Not found'));return
            origin=self.headers.get('Origin')
            if origin and origin not in [f'http://127.0.0.1:{port}',f'http://localhost:{port}']:
                self.reply(403,dict(error='Only local same-origin requests are accepted'));return
            if self.headers.get('Content-Type','').split(';')[0]!='application/json':self.reply(415,dict(error='Expected JSON'));return
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=MAX_REQUEST:self.reply(413,dict(error='Request must be below 8 MB'));return
                frames,roi=decode_request(self.rfile.read(length))
                with lock:result=predict_frames(model,bundle,frames,roi)
                self.reply(200,result)
            except (ValueError,TypeError,KeyError,OSError,Image.DecompressionBombError):
                self.reply(400,dict(error='Invalid images, JSON or mouth crop'))

        def log_message(self,format,*args):
            if args and str(args[0]).startswith('POST'):print('Prediction request processed',flush=True)
    return Handler


def serve(path,port=8765):
    model,bundle=load_bundle(path)
    server=ThreadingHTTPServer(('127.0.0.1',port),make_handler(model,bundle,port))
    print(f'Open http://127.0.0.1:{port} (loopback only)',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
