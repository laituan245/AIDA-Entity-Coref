# AIDA Entity Coreference

```
docker build --tag laituan245/spanbert_entity_coref .
docker push laituan245/spanbert_entity_coref
docker run --gpus '"device=0"' --rm -v /shared:/shared laituan245/spanbert_entity_coref -edl_official input1.tab -edl_freebase input2.tab -l ltf_dir -o output.tab
```
