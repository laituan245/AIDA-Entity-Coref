cp /shared/nas/data/m1/tuanml2/aida_entity_coref/pretrained/model.pt model.pt
docker build --tag laituan245/spanbert_entity_coref .
docker push laituan245/spanbert_entity_coref
