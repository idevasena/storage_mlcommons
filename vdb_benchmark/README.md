# Vector Database Benchmark Tool
This tool allows you to benchmark and compare the performance of vector databases with current support for Milvus and others planned.

## Installation

### Using Docker (recommended)
1. Clone the repository:
``` bash
git clone -b TF_VDBBench https://github.com/mlcommons/storage.git
cd storage/vdb_benchmark
```
2. Build and run the Docker container:
```bash
docker compose up -d # with docker-compose-v2. v1 uses docker-compose up
```

### Manual Installation
1. Clone the repository:
```bash
git clone -b TF_VDBBench https://github.com/mlcommons/storage.git
cd storage/vdb_benchmark
```

2. Install the package:
```bash
pip3 install ./
```

## Deploying a Standalone Milvus Instance
The docker-compose.yml file will configure a 3-container instance of Milvus database.
 - Milvus Database
 - Minio Object Storage
 - etcd 

The docker-compose.yml file uses ```/mnt/vdb``` as the root directory for the required docker volumes. You can modify the compose file for your environment or ensure that your target storage is mounted at this location.

For testing more than one storage solution, there are two methods:
1. Create a set of containers for each storage solution with modified docker-compose.yml files pointing to different root directories. Each set of containers will also need a different port to listen on. You may need to limit how many instances you can run depending on the available memory in your system
2. Bring down the containers, copy the /mnt/vdb data to another location, change the mount point to point to the new location. Bring the containers back up. This is simpler as the database connection isn't changing but you need to manually reconfigure the storage to change the system under test.

### Deployment
```bash
cd storage/vdb_benchmark
docker compose up -d # with docker-compose-v2. v1 uses docker-compose up
```

```-d``` option is required to detach from the containers after starting them. Without this option you will be attached to the log output of the set of containers and ```ctrl+c``` will stop the containers.

*If you have connection problems with a proxy I recommend this link: https://medium.com/@SrvZ/docker-proxy-and-my-struggles-a4fd6de21861*

## Running the Benchmark
The benchmark process consists of three main steps:
1. Loading vectors into the database
2. Monitoring and compacting the database
3. Running the benchmark queries

### Step 1: Load Vectors into the Database
Use the load_vdb.py script to generate and load 10 million vectors into your vector database: (this process can take up to 8 hours)

#### Default/ Standard Mode

##### Basic execution with config file
```bash
python vdbbench/load_vdb.py --config vdbbench/configs/10m_diskann.yaml
```

##### With explicit parameters (no config)
```bash
python vdbbench/load_vdb.py --collection-name benchmark_test \
    --dimension 1536 \
    --num-vectors 1000000 \
    --batch-size 10000
```

##### Override config values
```bash
python vdbbench/load_vdb.py --config vdbbench/configs/10m_diskann.yaml \
    --collection-name custom_collection \
    --num-vectors 500000 \
    --force
```

##### With reproducible seed
```bash
python vdbbench/load_vdb.py --config vdbbench/configs/10m_diskann.yaml \
    --seed 42
```

#### Adaptive Mode (Memory-Aware Batch Sizing)

##### Enable adaptive batching (auto-scales based on memory pressure)
```bash
python vdbbench/load_vdb.py --config vdbbench/configs/100m_diskann.yaml \
    --adaptive
```

##### With explicit memory budget
```bash
python vdbbench/load_vdb.py --config vdbbench/configs/100m_diskann.yaml \
    --adaptive \
    --memory-budget 4G
```
##### Adaptive with smaller budget for constrained systems
```bash
python vdbbench/load_vdb.py --config vdbbench/configs/100m_diskann.yaml \
    --adaptive \
    --memory-budget 2G \
    --batch-size 5000
```

#### Disk-Backed Mode (Billion-Scale / Low Memory)

##### Use memory-mapped temp file (default temp directory)
```bash
python vdbbench/load_vdb.py --config vdbbench/configs/1b_diskann.yaml \
    --disk-backed
```

##### Specify fast NVMe for temp storage
```bash
python vdbbench/load_vdb.py --config vdbbench/configs/1b_diskann.yaml \
    --disk-backed \
    --temp-dir /mnt/nvme/tmp
```

##### Disk-backed with seed for reproducibility
```bash
python vdbbench/load_vdb.py --config vdbbench/configs/1b_diskann.yaml \
    --disk-backed \
    --temp-dir /mnt/nvme/tmp \
    --seed 12345
```

For testing, we recommend using a smaller data by passing the num_vectors option:
```bash
python vdbbench/load_vdb.py --config vdbbench/configs/10m_diskann.yaml --collection-name mlps_500k_10shards_1536dim_uniform_diskann --num-vectors 500000
```

Key parameters:
* --collection-name: Name of the collection to create
* --dimension: Vector dimension
* --num-vectors: Number of vectors to generate
* --chunk-size: Number of vectors to generate in each chunk (for memory management)
* --distribution: Distribution for vector generation (uniform, normal)
* --batch-size: Batch size for insertion

Example configuration file (vdbbench/configs/10m_diskann.yaml):
```yaml
database:
  host: 127.0.0.1
  port: 19530
  database: milvus
  max_receive_message_length: 514_983_574
  max_send_message_length: 514_983_574

dataset:
  collection_name: mlps_10m_10shards_1536dim_uniform_diskann
  num_vectors: 10_000_000
  dimension: 1536
  distribution: uniform
  batch_size: 1000
  num_shards: 10
  vector_dtype: FLOAT_VECTOR

index:
  index_type: DISKANN
  metric_type: COSINE
  #index_params
  max_degree: 64
  search_list_size: 200

workflow:
  compact: True
```

### Step 2: Monitor and Compact the Database
The compact_and_watch.py script monitors the database and performs compaction. You should only need this if the load process exits out while waiting. The load script will do compaction and will wait for it to complete.
```bash
python vdbbench/compact_and_watch.py --config vdbbench/configs/10m_diskann.yaml --interval 5
```
This step is automatically performed at the end of the loading process if you set compact: true in your configuration.

### Step 3: Run the Benchmark
Finally, run the benchmark using the simple_bench.py script:
```bash
python vdbbench/simple_bench.py --host 127.0.0.1 --collection <collection_name> --processes <N> --batch-size <batch_size> --runtime <length of benchmark run in seconds>
```

For comparison with HNSW indexing, use ```vdbbench/configs/10m_hnsw.yaml``` and update collection_name accordingly.

## Supported Databases
Milvus with DiskANN & HNSW indexing (currently implemented)

# Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
