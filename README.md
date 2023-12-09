# GMM-EM and GMM-VB
<div align="center">
    <img
        src="images\EM_VB_4_clusters.png"
        alt="The data for clustering."
        title="The data for clustering."
        width=500px>
</div>
This directory publishes the examples of EM algorithm and variational Bayes (VB) that are applied to GMM.

# Description
In this implementation, four clusters will be automatically generated.
You can select the number of estimated clusters and the algorithms at `docker-compose.yml`.
- K: the number of clusters
- alg: EM or VB

# Usage
You can use the following command to run the demo.
```
git clone https://github.com/beginaid/GMM-EM-VB.git
cd GMM-EM-VB
docker compose up -d --build
```

Default settings are shown as below.
```
python src/main.py --K 4 --alg VB
```

After you can see the output, delete the container resources using the following command.
```
docker compose down
```

Note that you can change the parameters at docker-compose.yml and run the demo as many times as you like before the resources are down.
```
python main.py --K [The number of clusters] --alg [EM or VB]
```
