import argparse
import sys
from app.rag.pipeline import ClinicalAssistantOrchestrator

def main():
    parser = argparse.ArgumentParser(description="Query the Clinical Knowledge Assistant")
    parser.add_argument("--query", type=str, required=True, help="Clinical question to ask")
    parser.add_argument("--mode", type=str, default="hybrid", choices=["hybrid", "dense"], help="Retrieval mode")

    args = parser.parse_args()

    try:
        # Initialize the orchestrator (which includes routing and RAG)
        orchestrator = ClinicalAssistantOrchestrator(retriever_mode=args.mode)

        # Process the query
        response, results = orchestrator.process_query(args.query)

        # Output formatting
        print("\n" + "="*60)
        print(f" ROUTE: {response.route.upper()} ")
        print("="*60)
        print(f"\n{response.answer}\n")

        if response.route == "clinical_rag":
            print(f"Status: {response.evidence_status} | Clinician Review Required: {'YES' if response.clinician_review_required else 'NO'}")
            if response.caution:
                print(f"Caution: {response.caution}")

            print("\n--- Verified Citations ---")
            if not response.cited_chunk_ids:
                print("No specific chunks cited.")
            else:
                # Resolve cited_chunk_ids to metadata
                for cid in response.cited_chunk_ids:
                    # Find the result with this ID
                    res = next((r for r in results if r['chunk_id'] == cid), None)
                    if res:
                        meta = res.get('metadata', {})
                        file = meta.get('source_file', 'Unknown')
                        page = meta.get('page_start', 'Unknown')
                        section = meta.get('section_path', 'Unknown')
                        print(f"- {file} (Page {page}, Section: {section})")
                    else:
                        print(f"- ID {cid}: [Citations not found in retrieved evidence]")

        print("\n" + "="*60)
        m = response.metrics
        print(f"Latency: Routing: {m.get('routing_ms', 0):.2f}ms | Retrieval: {m.get('retrieval_ms', 0):.2f}ms | Generation: {m.get('generation_ms', 0):.2f}ms | Total: {m.get('total_ms', 0):.2f}ms")
        print("="*60 + "\n")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
