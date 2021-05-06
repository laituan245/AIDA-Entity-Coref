# AIDA Entity Coreference

```
docker build --tag laituan245/spanbert_entity_coref:no_gpu .
docker push laituan245/spanbert_entity_coref:no_gpu
docker run --gpus '"device=0"' --rm -v /shared:/shared laituan245/spanbert_entity_coref:no_gpu -edl_official input1.tab -edl_freebase input2.tab -l ltf_dir -o output.tab
```
