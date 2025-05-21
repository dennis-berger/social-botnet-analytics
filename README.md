# social-botnet-analytics

A Social Media Analytics project for detecting and analyzing bot networks on Twitter‐derived data. We combine community‐detection (Louvain + planned k r-Clique), network‐science metrics, and a Linear-Threshold model to quantify each bot cluster’s “outer influence” on human audiences.

---


## Prerequisites

* Python 3.9+
* [Neo4j](https://neo4j.com/)

---

## Installation & Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/dennis-berger/social-botnet-analytics.git
   cd social-botnet-analytics
   ```

2. **Create & activate a virtual environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Python dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure file paths & Neo4j**

   * Edit `mgtab_root:` to point at `data/raw/`
   * Under `neo4j:` set `uri`, `user`, `password` for your database

---

## Running the Notebooks

Execute the notebooks in order to reproduce the entire pipeline:

1. **00_environment_setup.ipynb**  
   - Verify directory layout  
   - Process raw MGTAB “.pt” files into `data/processed/cleaned_data.pt` via `src/data/mgtab_dataset.py`

2. **01_data_cleaning.ipynb**  
   - Load the cleaned data  
   - Remove self‐loops, coalesce duplicate edges  
   - Inspect feature matrix (NaNs, duplicates) and compute train/val/test splits

3. **02_graph_analysis_and_anomaly_detection.ipynb**  
   - Build a NetworkX graph from PyG data  
   - Compute degree, centrality, connectivity measures  
   - Detect anomalous nodes (e.g. super‐hubs)

4. **03_community_detection_and_influence_modeling.ipynb**  
   - Run Louvain clustering (and outline kr-Clique/C-Tree)  
   - Load clusters into Neo4j  
   - Execute Cypher for bot/human profiling  
   - Simulate the Linear-Threshold model to compute outer‐influence

5. **04_social_graph_mining.ipynb**  
   - Explore advanced graph‐mining topics from the lecture (e.g. assortativity, link‐prediction)

6. **05_Static_Interactive_Graph.ipynb**  
   - Generate static Matplotlib plots 
   - Build simple interactive demos (vis.js/D3) for web‐based exploration

7. **06_graph_persistence.ipynb**  
   - Export `nodes.csv`, `edges.csv`, `clusters.csv`  
   - Persist the graph in Neo4j using py2neo or `LOAD CSV`

8. **07_visualization.ipynb**  
   - Consolidate and refine all report‐quality figures  
   - Tweak styles, labels, and layouts for final inclusion in the write-up

---

## Neo4j Import

After generating CSVs, load data into Neo4j with these commands:

```cypher
// 1) Create uniqueness constraint
CREATE CONSTRAINT ON (u:User) ASSERT u.user_id IS UNIQUE;

// 2) Load user nodes
LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row
CREATE (:User {
  user_id: toInteger(row.user_id),
  is_bot: CASE row.is_bot WHEN '1' THEN true ELSE false END,
  stance: toInteger(row.stance),
  train_mask: row.train_mask = 'True',
  val_mask:   row.val_mask   = 'True',
  test_mask:  row.test_mask  = 'True'
});

// 3) Load cluster labels
LOAD CSV WITH HEADERS FROM 'file:///clusters.csv' AS row
MATCH (u:User {user_id: toInteger(row.user_id)})
SET u.cluster = toInteger(row.cluster);

// 4) Load follow relationships
LOAD CSV FROM 'file:///edges.csv' AS line
WITH split(line, ',') AS cols
MATCH (a:User {user_id: toInteger(cols[0])})
MATCH (b:User {user_id: toInteger(cols[1])})
MERGE (a)-[:FOLLOWS]->(b);
```

---

## Report

* All analysis, figures, and discussion are in the Jupyter notebooks.
* The final written report (PDF) references these notebooks and includes embedded figures.

## Dataset
We use the MGTAB dataset available at https://github.com/GraphDetec/MGTAB/tree/main
