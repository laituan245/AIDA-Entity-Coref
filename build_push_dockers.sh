cp /shared/nas/data/m1/tuanml/coref/trained_model/chinese_entity_coref.pt model.pt
docker build --tag laituan245/chinese_entity_coref .
docker push laituan245/chinese_entity_coref
