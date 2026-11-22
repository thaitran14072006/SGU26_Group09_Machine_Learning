"""Bundle integration, malformed CSV validation and HTTP inference parity."""
import io
import json
from pathlib import Path
import sys
import threading
import unittest
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from http.server import ThreadingHTTPServer

import pandas as pd
from wesad.inference import DEFAULT_BUNDLE, predict_text, read_input
from wesad.utils import PROJECT_ROOT
sys.path.insert(0,str(PROJECT_ROOT/'app'))
from server import Handler


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text=(PROJECT_ROOT/'app/examples/features.csv').read_text(encoding='utf8')
        cls.frame=pd.read_csv(io.StringIO(cls.text))
        cls.columns=json.loads((DEFAULT_BUNDLE/'metadata.json').read_text())['feature_columns']

    def test_saved_examples_and_reordered_columns(self):
        for task in ['binary','multiclass']:
            expected=pd.read_csv(PROJECT_ROOT/f'app/examples/expected_{task}.csv')
            actual=predict_text(self.text,task)
            pd.testing.assert_frame_equal(actual,expected)
            pd.testing.assert_frame_equal(actual,predict_text(self.frame[self.frame.columns[::-1]].to_csv(index=False),task))

    def test_reject_missing_duplicate_empty_and_infinite(self):
        invalid=[self.frame.drop(columns=self.columns[0]).to_csv(index=False),
                 self.text.splitlines()[0]+'\n',
                 self.text.replace(self.columns[1],self.columns[0],1)]
        for value in ['invalid','inf']:
            f=self.frame.copy();f[self.columns[0]]=value;invalid.append(f.to_csv(index=False))
        for text in invalid:
            with self.subTest(text=text[:50]), self.assertRaises(ValueError):
                read_input(text,self.columns)

    def test_partial_missing_imputed_but_empty_row_rejected(self):
        f=self.frame.copy();f.loc[0,self.columns[0]]=float('nan')
        self.assertEqual(len(predict_text(f.to_csv(index=False))),len(f))
        f.loc[0,self.columns]=float('nan')
        with self.assertRaises(ValueError):read_input(f.to_csv(index=False),self.columns)

    def test_http_upload_and_validation(self):
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        url=f'http://127.0.0.1:{server.server_port}'
        try:
            with urlopen(url) as response:self.assertIn('WESAD',response.read().decode())
            for task in ['binary','multiclass']:
                req=Request(url+'/predict?task='+task,data=self.text.encode(),headers={'Content-Type':'text/csv'})
                with urlopen(req) as response:body=json.load(response)
                self.assertEqual(body['rows'],predict_text(self.text,task).to_dict(orient='records'))
                self.assertEqual(sum(body['counts'].values()),30)
            for text in ['bad\n1\n','']:
                with self.assertRaises(HTTPError) as exc:
                    urlopen(Request(url+'/predict',data=text.encode()))
                self.assertEqual(exc.exception.code,400)
        finally:
            server.shutdown();server.server_close();thread.join()


if __name__=='__main__':unittest.main()
