import { serve } from '@hono/node-server'
import { Hono } from 'hono'
import path from "path";
import { promises as fs } from "fs";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { OllamaEmbeddings } from "@langchain/community/embeddings/ollama";
import {PromptTemplate} from "@langchain/core/prompts";
import {createStuffDocumentsChain} from "langchain/chains/combine_documents";
import {Ollama} from "@langchain/community/llms/ollama";

const app = new Hono()

const ollama = new Ollama({
  baseUrl: "http://localhost:11434", // Default value
  model: "llama3.1:latest", // gemma2:2b
});

const getTextFile = async () => {

  const filePath = path.join(__dirname, "../data/langchain-test.txt");

  const data = await fs.readFile(filePath, "utf-8");

  return data;
}

app.get('/', (c) => {
  return c.text('Hello Hono!')
})

// Vector Db

let vectorStore : MemoryVectorStore;

app.get('/loadTextEmbeddings', async (c) => {

  const text = await getTextFile();

    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      separators:['\n\n', '\n', ' ', '', '###'],
      chunkOverlap: 50
    });

    const output = await splitter.createDocuments([text])

  const embeddings = new OllamaEmbeddings({
    model: "llama3.1:latest", // gemma2:2b
    baseUrl: "http://localhost:11434", // default value
    requestOptions: {
      useMMap: true, // use_mmap 1
      numThread: 6, // num_thread 6
      numGpu: 1, // num_gpu 1
    },
  });

    vectorStore = await MemoryVectorStore.fromDocuments(output, embeddings);

    const response = {message: "Text embeddings loaded successfully."};

    return c.json(response);
})

app.post('/ask',async (c) => {

  const { question } = await c.req.json();

  if(!vectorStore){
    return c.json({message: "Text embeddings not loaded yet."});
  }

  const prompt = PromptTemplate.fromTemplate(`You are a helpful AI assistant. Answer the following question based only on the provided context. If the answer cannot be derived from the context, say "I don't have enough information to answer that question." If I like your results I'll tip you $1000!

Context: {context}

Question: {question}

Answer: 
  `)

});

const documentChain = await createStuffDocumentsChain()

const port = 3002
console.log(`Server is running on port ${port}`)

serve({
  fetch: app.fetch,
  port
})
