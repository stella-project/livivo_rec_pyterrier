import re
import jsonlines
import pandas as pd
import unidecode
import pyterrier as pt
if not pt.started():
  pt.init()




def _livivo_doc_iter():
    with jsonlines.open('./data/livivo/publications/publications.jsonl') as reader:
        for obj in reader:
            title = obj.get('TITLE') or ''
            title = title[0] if type(title) is list else title
            abstract = obj.get('ABSTRACT') or ''
            abstract = abstract[0] if type(abstract) is list else abstract
            yield {'docno': obj.get('DBRECORDID'), 'text': ' '.join([title, abstract])}


class Ranker(object):

    def __init__(self):
        self.idx = None

    def index(self):
        pass

    def rank_publications(self, query, page, rpp):

        itemlist = []

        return {
            'page': page,
            'rpp': rpp,
            'query': query,
            'itemlist': itemlist,
            'num_found': len(itemlist)
        }


class Recommender(object):

    def __init__(self):
        self.idx = None
        self.title_lookup = {}

    def index(self):
        iter_indexer = pt.IterDictIndexer("./index")
        doc_iter = _livivo_doc_iter()
        indexref = iter_indexer.index(doc_iter)
        self.idx = pt.IndexFactory.of(indexref)

        with jsonlines.open('./data/livivo/publications/publications.jsonl') as reader:
            for obj in reader:
                title = obj.get('TITLE') or ''
                title = title[0] if type(title) is list else title
                self.title_lookup[obj.get('DBRECORDID')] = title

        print(f'# of elems in lookup list: {len(self.title_lookup)}')

    def recommend_publications(self, item_id, page, rpp):

        itemlist = []

        doc_title = self.title_lookup.get(item_id)
        
        if (doc_title):
            doc_title = re.sub(r'[^\w\s]', ' ', doc_title)

            doc_title = unidecode.unidecode(doc_title)

            if doc_title is not None:
                topics = pd.DataFrame.from_dict({'qid': [0], 'query': [doc_title]})
                retr = pt.BatchRetrieve(self.idx, controls={"wmodel": "TF_IDF"})
                retr.setControl("wmodel", "TF_IDF")
                retr.setControls({"wmodel": "TF_IDF"})
                res = retr.transform(topics)
                itemlist = list(res['docno'][page*rpp:(page+1)*rpp])
            
        return {
            'page': page,
            'rpp': rpp,
            'item_id': item_id,
            'itemlist': itemlist,
            'num_found': len(itemlist)
        }

    def recommend_datasets(self, item_id, page, rpp):

        itemlist = []

        return {
            'page': page,
            'rpp': rpp,
            'item_id': item_id,
            'itemlist': itemlist,
            'num_found': len(itemlist)
        }
