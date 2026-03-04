import glob

for path in glob.glob('src/director_ai/integrations/*.py'):
    with open(path, 'r', encoding='utf-8') as f:
        c = f.read()

    c = c.replace('asyncio.run(self.scorer.review(query.strip()), claim.strip())', 'asyncio.run(self.scorer.review(query.strip(), claim.strip()))')
    c = c.replace('asyncio.run(self.scorer.review(query), response)', 'asyncio.run(self.scorer.review(query, response))')
    c = c.replace('self.store.add(k, v)', 'asyncio.run(self.store.add(k, v))')
    c = c.replace('self.store.add(doc_id=doc.id, text=doc.text)', 'asyncio.run(self.store.add(doc_id=doc.id, text=doc.text))')
    c = c.replace('self.store.add(d[\"id\"], d[\"text\"])', 'asyncio.run(self.store.add(d[\"id\"], d[\"text\"]))')
    c = c.replace('self.store.add(node.node_id, text)', 'asyncio.run(self.store.add(node.node_id, text))')
    c = c.replace('self.store.add(doc_id, text)', 'asyncio.run(self.store.add(doc_id, text))')
    c = c.replace('self.store.add(doc.metadata.get(\"id\", \"\"), doc.page_content)', 'asyncio.run(self.store.add(doc.metadata.get(\"id\", \"\"), doc.page_content))')
    c = c.replace('self.store.add(doc.id, doc.content)', 'asyncio.run(self.store.add(doc.id, doc.content))')

    with open(path, 'w', encoding='utf-8') as f:
        f.write(c)
