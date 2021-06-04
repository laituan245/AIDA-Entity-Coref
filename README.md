# AIDA Entity Coreference

```
docker build --tag laituan245/spanbert_entity_coref:no_el .
docker push laituan245/spanbert_entity_coref:no_el
docker run --gpus '"device=0"' --rm -v /shared:/shared laituan245/spanbert_entity_coref:no_el -edl_official input1.tab -edl_freebase input2.tab -l ltf_dir -o output.tab
```
