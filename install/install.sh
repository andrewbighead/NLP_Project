wget https://github.com/milvus-io/milvus/releases/download/v2.2.12/milvus-standalone-docker-compose.yml -O docker-compose.yml
sudo docker compose up -d
docker run --name Neo4JNLP -d --publish=7474:7474 --publish=7687:7687 --env NEO4J_AUTH=neo4j/marioandrea neo4j:5.9.0
git clone https://github.com/andrewbighead/NLP_Project.git
virtualenv nlp
source nlp/bin/activate
pip install numpy
pip install torch
pip install soundfile
pip install librosa
pip install tqdm
pip install datasets
pip install speechbrain
pip install transformers
pip install neo4j
pip install pymilvus
pip install py2neo
cd NLP_Project/code
python3 install.py
