<h1>RFCKG</h1>

<h2>1. Pretraining IoT BERT</h2>
Run "src/prepare_pretrain_data.py" to prepare the data for pretraining BERT.

Run "src/pretrain_iot_BERT.py" to pretrain BERT on Masked Language Modelling and Next Sentence Prediction. This is
following the approach described in the following video: https://www.youtube.com/watch?v=IC9FaVPKlYc&t=1079s.

<h2>2. Fine-tuning IoT BERT for entity extraction</h2>
Run "src/entity_extractor.py" to use IoT BERT and fine tune it on the entity extraction task on CoAP protocol only. We
load IoT BERT and stack a token classification layer on top of it.

Run "src/extract_MQTT_entity.py" to test the generalisation ability.

<h2>3. Fine-tuning IoT BERT for relation extraction</h2>
Run "src/relation_extractor.py" to use IoT BERT and fine tune it on the relation extraction task. We load IoT BERT and
stack a sequence classification layer on top of it. This is still in experiment.