import os
import sys
import argparse
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.loader import load_resumes
from src.retriever import ResumeRetriever
from src.agent import ResumeScreeningAgent

def main():
    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is missing. Set it in .env or your shell.")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="TalentGraph: Agentic Resume Screening Engine")
    parser.add_argument(
        "--query",
        type=str,
        default="Which candidate has hands-on automated testing experience with Selenium?",
        help="Recruiter query to evaluate"
    )
    parser.add_argument(
        "--resumes_dir",
        type=str,
        default="resumes",
        help="Directory containing candidate resume PDFs"
    )
    args = parser.parse_args()

    print(f"Loading candidate resumes from '{args.resumes_dir}'...")
    resumes = load_resumes(args.resumes_dir)
    print(f"Successfully loaded {len(resumes)} resumes.")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    print("Initializing multi-stage retriever & vector index...")
    retriever = ResumeRetriever(resumes=resumes, embeddings=embeddings, llm=llm)

    print("Assembling LangGraph Agentic RAG workflow...")
    agent = ResumeScreeningAgent(retriever=retriever, llm=llm)

    print(f"\nEvaluating Query: '{args.query}'\n" + "="*60)
    result = agent.run(args.query)

    if result.get("direct_response"):
        print("\nResponse:")
        print(result["direct_response"])
        return

    report = result.get("report")
    if report:
        print("\n================ FINAL SCREENING REPORT ================")
        print(f"Executive Summary: {report.summary}\n")
        for c in report.candidates:
            print(f"• Candidate: {c.name} (Match Score: {c.match_score}/10)")
            print(f"  Key Evidence: {c.evidence}")
            print(f"  Identified Gaps/Unknowns: {c.unknowns}\n")

if __name__ == "__main__":
    main()
