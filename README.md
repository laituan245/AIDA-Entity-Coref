# AIDA Entity Coreference

```
docker build --tag laituan245/merge_ru_linking .
docker push laituan245/merge_ru_linking
docker run --gpus '"device=0"' --rm -v /shared:/shared laituan245/merge_ru_linking -edl_official input1.tab -edl_freebase input2.tab -o output.tab
```
