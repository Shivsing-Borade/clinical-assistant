import argparse
import sys
from app.rag.pipeline import ClinicalRAGPipeline

def main():
    parser = argparse.ArgumentParser(description="Query the Clinical Knowledge Assistant")
    parser.add_argument("--query", type=str, required=True, help="Clinical question to ask")
    parser.add_argument("--mode", type=str, default="hybrid", choices=["hybrid", "dense"], help="Retrieval mode")

    args = parser.parse_args()

    try:
        # Initialize the pipeline
        pipeline = ClinicalRAGPipeline(retriever_mode=args.mode)

        # Execute the pipeline
        answer, results, metrics = pipeline.ask(args.query)

        # Output formatting
        print("\n" + "="*60)
        print(" CLINICAL ANSWER ")
        print("="*60)
        print(f"\n{answer.answer}\n")
        print(f"Status: {answer.evidence_status} | Clinician Review Required: {'YES' if answer.clinician_review_required else 'NO'}")
        if answer.caution:
            print(f"Caution: {answer.caution}")

        print("\n--- Verified Citations ---")
        if not answer.cited_chunk_ids:
            print("No specific chunks cited.")
        else:
            # Resolve cited_chunk_ids to metadata
            for cid in answer.cited_chunk_ids:
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
        print(f"Latency: Retrieval: {metrics['retrieval_ms']:.2f}ms | Generation: {metrics['generation_ms']:.2f}ms | Total: {metrics['total_ms']:.2f}ms")
        print("="*60 + "\n")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
