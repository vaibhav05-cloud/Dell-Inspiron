from graph.spacy_extractor import extract_entities_spacy
from graph.tech_extractor import extract_technical_entities
from graph.entity_merger import merge_entities


class HybridEntityExtractor:

    def extract_all(self, chunks):

        all_entities = []

        for chunk in chunks:

            content = chunk.get("content", "")

            spacy_entities = extract_entities_spacy(content)

            tech_entities = extract_technical_entities(content)

            entities = merge_entities(
                spacy_entities,
                tech_entities
            )

            for entity in entities:

                entity["chunk_id"] = chunk["chunk_id"]
                entity["page_number"] = chunk.get("page_number")
                entity["section_name"] = chunk.get("section_name", "")
                entity["source_file"] = chunk.get("source_file", "")

            all_entities.extend(entities)

        return all_entities