from sentence_transformers import SentenceTransformer, models

model_name = "/root/Agent/e5-base-v2"
word_embedding_model = models.Transformer(model_name)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
model.save(model_name + "_sbert")
