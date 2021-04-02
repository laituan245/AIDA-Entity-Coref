cp /shared/nas/data/m1/tuanml/coref/trained_model/cn_en_entity_coref.pt model.pt
docker build --tag laituan245/cn_en_entity_coref .
docker push laituan245/cn_en_entity_coref
