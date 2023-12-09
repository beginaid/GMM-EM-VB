# GMM-EM and GMM-VB
<div align="center">
    <img
        src="images\EM_VB_4_clusters.png"
        alt="The data for clustering."
        title="The data for clustering."
        width=500px>
</div>
This directory publishes the examples of EM algorithms and variational Bayes (VB) that is applied to GMM.

# Description
In this implementation, four clusters will be automatically generated for clustering.
Users can select the number of estimated clusters and the algorithms.

# Usage
You can use the following command to run the demo.
```
git clone https://github.com/beginaid/GMM-EM-VB.git
cd GMM-EM-VB
docker compose up -d --build
```

After you can see these outputs, delete the container resources using the following command.
```
docker compose down
```
Note that you can change the parameters in docker-compose.yml and run the demo as many times as you like before the resources are down.
```
python main.py --K [The number of clusters] --alg [EM or VB]
```