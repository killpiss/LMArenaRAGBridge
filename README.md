# LMArenaRAGBridge
A reverse-engineered bridge for LMArena using RAG Storage to get around context limits.

# HOW TO USE:

1. Put the .js file to tampermonkey and enable it.
2. Run this upon opening the terminal: 

pip install -r requirements.txt

3. Run the python script.
4. Open https://lmarena.ai/
5. Terminal should say "Updated model registry with [x] models."
6. Endpoint should be: http://127.0.0.1:9080/v1 or http://localhost:9080/v1

(PLEASE READ) GENERAL TROUBLESHOOTING:

You may have an occasional really long wait for a response, cancel the message if you notice this and it should work fine again upon generating a new message. To spot this, I reccomend having Text Streaming on.

Make sure you have the "chroma_db" folder and that it isn't corrupted you DINGUS!

In LMArena-Hybrid.py the line: "MAX_PROMPT_TOKENS_API = 31000" controls the amount of tokens until RAG activates.

The line: "B_RAG =" controls the RAG budget, higher equals more.
