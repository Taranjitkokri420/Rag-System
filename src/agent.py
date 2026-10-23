from typing import Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from .schemas import AgentState, QueryRoute, RetrievalGrade, ScreeningReport
from .retriever import ResumeRetriever

class ResumeScreeningAgent:
    """Autonomous self-correcting agent for recruiter inquiry handling and candidate evaluation."""
    def __init__(self, retriever: ResumeRetriever, llm: ChatOpenAI):
        self.retriever = retriever
        self.llm = llm
        self.router_llm = self.llm.with_structured_output(QueryRoute)
        self.grader_llm = self.llm.with_structured_output(RetrievalGrade)
        self.structured_llm = self.llm.with_structured_output(ScreeningReport)
        self.app = self._build_graph()

    def route_query(self, query: str) -> QueryRoute:
        prompt = (
            "Classify the following recruiter query into one of four routing strategies:\n"
            "1. single_candidate_lookup: targeting a specific named candidate.\n"
            "2. candidate_comparison: comparing two or more specific named candidates.\n"
            "3. pool_screening: searching the candidate pool for specific skills, roles, or qualifications.\n"
            "4. out_of_scope: unrelated to hiring, resumes, or candidates.\n\n"
            f"Query: {query}"
        )
        return self.router_llm.invoke(prompt)

    def grade_context(self, query: str, context: str) -> RetrievalGrade:
        prompt = (
            f"Recruiter Query: {query}\n\n"
            f"Retrieved Context:\n{context}\n\n"
            "Determine if the retrieved context contains sufficient, relevant facts to answer the recruiter's question. "
            "Grade strictly based on factual presence, not general plausibility."
        )
        return self.grader_llm.invoke(prompt)

    def rewrite_query(self, query: str, critique: str) -> str:
        prompt = (
            f"Original Recruiter Query: {query}\n"
            f"Critique of Initial Retrieval: {critique}\n\n"
            "Formulate a more specific, keyword-dense search query targeting the missing technical skills and candidate qualifications."
        )
        return self.llm.invoke(prompt).content.strip()

    def _build_graph(self):
        def router_node(state: AgentState):
            decision = self.route_query(state["query"])
            return {
                "route": decision.route,
                "target_candidates": decision.candidate_names,
                "current_search_query": decision.cleaned_query,
                "rewrite_attempts": 0
            }

        def retrieve_node(state: AgentState):
            route = state["route"]
            # Target optimization for named candidate
            if route == "single_candidate_lookup" and state.get("target_candidates"):
                matched_cids = [
                    cid for cid, info in self.retriever.resumes.items()
                    if any(name.lower() in info["name"].lower() for name in state["target_candidates"])
                ]
                if matched_cids:
                    context = "\n\n".join(
                        f"--- {self.retriever.resumes[cid]['name']} ({self.retriever.resumes[cid]['role']}) ---\n{self.retriever.resumes[cid]['text']}"
                        for cid in matched_cids
                    )
                    return {"context": context}

            context, _ = self.retriever.small_to_big_retrieval(state["current_search_query"], k=4, fetch_k=15)
            return {"context": context}

        def grade_node(state: AgentState):
            grade = self.grade_context(state["query"], state["context"])
            return {"retrieval_grade": grade}

        def rewrite_node(state: AgentState):
            attempts = state.get("rewrite_attempts", 0) + 1
            critique = state["retrieval_grade"].explanation if state.get("retrieval_grade") else "Low relevance score"
            new_query = self.rewrite_query(state["query"], critique)
            return {
                "current_search_query": new_query,
                "rewrite_attempts": attempts
            }

        def generate_node(state: AgentState):
            prompt = (
                "You are an expert recruitment assistant. Answer the recruiter's inquiry strictly using "
                "the candidate resumes provided below. If information is absent, explicitly record it in the 'unknowns' field.\n\n"
                f"Context:\n{state['context']}\n\nRecruiter Query: {state['query']}"
            )
            report = self.structured_llm.invoke(prompt)
            return {"report": report}

        def out_of_scope_node(state: AgentState):
            response = (
                "This system is specialized for candidate resume screening and technical evaluation. "
                "The requested query is outside the scope of the recruitment applicant pool."
            )
            return {"direct_response": response}

        def route_after_classifier(state: AgentState):
            if state["route"] == "out_of_scope":
                return "out_of_scope_node"
            return "retrieve_node"

        def route_after_grader(state: AgentState):
            grade = state.get("retrieval_grade")
            attempts = state.get("rewrite_attempts", 0)
            if grade and not grade.is_sufficient and attempts < 1:
                return "rewrite_node"
            return "generate_node"

        workflow = StateGraph(AgentState)
        workflow.add_node("router_node", router_node)
        workflow.add_node("retrieve_node", retrieve_node)
        workflow.add_node("grade_node", grade_node)
        workflow.add_node("rewrite_node", rewrite_node)
        workflow.add_node("generate_node", generate_node)
        workflow.add_node("out_of_scope_node", out_of_scope_node)

        workflow.set_entry_point("router_node")
        workflow.add_conditional_edges("router_node", route_after_classifier)
        workflow.add_edge("retrieve_node", "grade_node")
        workflow.add_conditional_edges("grade_node", route_after_grader)
        workflow.add_edge("rewrite_node", "retrieve_node")
        workflow.add_edge("generate_node", END)
        workflow.add_edge("out_of_scope_node", END)

        return workflow.compile()

    def run(self, query: str) -> Dict[str, Any]:
        initial_state = {
            "query": query,
            "current_search_query": query,
            "route": "",
            "target_candidates": [],
            "context": "",
            "retrieval_grade": None,
            "rewrite_attempts": 0,
            "report": None,
            "direct_response": None
        }
        return self.app.invoke(initial_state)
