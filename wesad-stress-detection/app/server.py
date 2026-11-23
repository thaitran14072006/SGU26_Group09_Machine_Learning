"""Local CSV demo for the exported LR candidate: python app/server.py."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
from urllib.parse import urlparse, parse_qs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from wesad.inference import predict_text


class Handler(BaseHTTPRequestHandler):
    def reply(self, status, body, kind='application/json; charset=utf-8'):
        content = body.encode('utf8')
        self.send_response(status)
        self.send_header('Content-Type',kind)
        self.send_header('Content-Length',str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == '/':
            self.reply(200,(ROOT/'app/index.html').read_text(encoding='utf8'),'text/html; charset=utf-8')
        elif path == '/example.csv':
            self.reply(200,(ROOT/'app/examples/features.csv').read_text(encoding='utf8'),'text/csv; charset=utf-8')
        else:
            self.reply(404,json.dumps({'error':'Not found'}))

    def do_POST(self):
        url = urlparse(self.path)
        if url.path != '/predict':
            self.reply(404,json.dumps({'error':'Not found'})); return
        try:
            size = int(self.headers.get('Content-Length','0'))
            if not 0 < size <= 5*1024*1024:
                raise ValueError('Upload a nonempty CSV file smaller than 5 MiB.')
            text = self.rfile.read(size).decode('utf-8-sig')
            task = parse_qs(url.query).get('task',['multiclass'])[0]
            result = predict_text(text,task)
            self.reply(200,json.dumps({'rows':result.to_dict(orient='records'),
                                      'counts':result.predicted_name.value_counts().to_dict()},ensure_ascii=False))
        except (ValueError,UnicodeError) as exc:
            self.reply(400,json.dumps({'error':str(exc)},ensure_ascii=False))


if __name__ == '__main__':
    print('LR CSV demo: http://127.0.0.1:8080',flush=True)
    ThreadingHTTPServer(('127.0.0.1',8080),Handler).serve_forever()
