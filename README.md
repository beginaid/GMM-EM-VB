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
python main.py --K [The number of clusters] --alg [EM or VB]
```


# Requirement
- Python: 3.7.12
- fire: 0.40
- matplotlib: 3.2.2
- numpy: 1.19.5