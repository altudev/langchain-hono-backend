import { serve } from '@hono/node-server'
import { Hono } from 'hono'
import path from "path";
import { promises as fs } from "fs";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

const app = new Hono()

const getTextFile = async () => {

  const filePath = path.join(__dirname, "../data/langchain-test.txt");

  const data = await fs.readFile(filePath, "utf-8");

  return data;
}

app.get('/', (c) => {
  return c.text('Hello Hono!')
})

app.get('/loadTextEmbeddings', async (c) => {

  const text = await getTextFile();

    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      separators:['\n\n', '\n', ' ', '', '###'],
      chunkOverlap: 50
    });
})

const port = 3002
console.log(`Server is running on port ${port}`)

serve({
  fetch: app.fetch,
  port
})
