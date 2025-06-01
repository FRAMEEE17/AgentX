
from typing import Literal, Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
import operator
class LeanState(TypedDict):
    """State schema matching your current workflow"""
    problem_description: str
    function_name: str
    problem_type: str
    complexity_score: float
    generated_code: str
    generated_proof: str
    iteration_count: int
    verification_result: bool
    cache_hit: bool
    reasoning_trace: Annotated[list, operator.add]

def problem_analysis(state: LeanState):
    """🔍 Problem Analysis - Extract function signature and complexity"""
    return {"problem_type": "analyzed", "reasoning_trace": ["Analysis complete"]}

def rag_query(state: LeanState):
    """📚 RAG Query - Retrieve relevant examples"""
    return {"reasoning_trace": ["RAG query complete"]}

def generation(state: LeanState):
    """⚡ Code Generation - Generate implementation and proof"""
    return {"generated_code": "x", "generated_proof": "rfl"}

def verification(state: LeanState):
    """✅ Verification - Execute Lean code"""
    return {"verification_result": True}

def cache_store(state: LeanState):
    """💾 Cache Solution - Store successful patterns"""
    return {"cache_hit": True}

def iteration_increment(state: LeanState):
    """🔄 Iteration Increment - Handle retry logic"""
    return {"iteration_count": state["iteration_count"] + 1}

# =============================================================================
# ROUTING FUNCTIONS
# =============================================================================

def success_check(state: LeanState) -> Literal["cache_store", "iteration_check"]:
    """Route based on verification success"""
    if state["verification_result"] and state["generated_code"] != "sorry":
        return "cache_store"
    return "iteration_check"

def iteration_check(state: LeanState) -> Literal["rag_query", END]:
    """Route based on iteration limit"""
    if state["iteration_count"] < 3:
        return "rag_query"
    return END

# =============================================================================
# BUILD YOUR EXACT WORKFLOW GRAPH
# =============================================================================

def create_rag_cag_graph():
    """Create the exact graph structure from your workflow"""
    
    # Initialize StateGraph
    builder = StateGraph(LeanState)
    
    # Add nodes (matching your workflow)
    builder.add_node("problem_analysis", problem_analysis)
    builder.add_node("rag_query", rag_query) 
    builder.add_node("generation", generation)
    builder.add_node("verification", verification)
    builder.add_node("cache_store", cache_store)
    builder.add_node("iteration_increment", iteration_increment)
    
    # Add edges (matching your flow)
    builder.add_edge(START, "problem_analysis")
    builder.add_edge("problem_analysis", "rag_query")
    builder.add_edge("rag_query", "generation")
    builder.add_edge("generation", "verification")
    
    # Add conditional edges
    builder.add_conditional_edges(
        "verification", 
        success_check,
        {
            "cache_store": "cache_store",
            "iteration_check": "iteration_increment"
        }
    )
    
    # Terminal edges
    builder.add_edge("cache_store", END)
    builder.add_conditional_edges(
        "iteration_increment",
        iteration_check,
        {
            "rag_query": "rag_query", 
            END: END
        }
    )
    
    return builder.compile()

# =============================================================================
# GENERATE DIAGRAMS USING LANGGRAPH BUILT-INS
# =============================================================================

def generate_mermaid_code():
    """Generate Mermaid code using LangGraph's built-in method"""
    graph = create_rag_cag_graph()
    
    # Use LangGraph's built-in mermaid generation
    mermaid_code = graph.get_graph().draw_mermaid()
    
    print("🎨 Generated Mermaid Code (LangGraph built-in):")
    print("=" * 60)
    print(mermaid_code)
    print("=" * 60)
    
    # Save to file
    with open("rag_cag_langgraph.mmd", "w") as f:
        f.write(mermaid_code)
    
    print("💾 Saved to: rag_cag_langgraph.mmd")
    return mermaid_code

def generate_png_diagram():
    """Generate PNG using LangGraph's built-in method"""
    graph = create_rag_cag_graph()
    
    try:
        # Use LangGraph's built-in PNG generation
        png_data = graph.get_graph().draw_mermaid_png()
        
        # Save PNG file
        with open("rag_cag_workflow.png", "wb") as f:
            f.write(png_data)
        
        print("🖼️  PNG diagram saved to: rag_cag_workflow.png")
        return True
        
    except Exception as e:
        print(f"❌ PNG generation failed: {e}")
        print("💡 Try: pip install pyppeteer")
        return False

def display_graph_info():
    """Display graph information"""
    graph = create_rag_cag_graph()
    graph_info = graph.get_graph()
    
    print("\n📊 Graph Information:")
    print(f"   Nodes: {len(graph_info.nodes)}")
    print(f"   Edges: {len(graph_info.edges)}")
    
    print("\n🔗 Nodes:")
    for node_id in graph_info.nodes:
        print(f"   - {node_id}")
    
    print("\n➡️  Edges:")
    for edge in graph_info.edges:
        print(f"   - {edge.source} → {edge.target}")

# =============================================================================
# ADVANCED VISUALIZATION OPTIONS
# =============================================================================

def generate_with_custom_styling():
    """Generate with custom styling options from LangGraph docs"""
    graph = create_rag_cag_graph()
    
    try:
        from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
        
        # Custom styling
        png_data = graph.get_graph().draw_mermaid_png(
            curve_style=CurveStyle.LINEAR,
            node_colors=NodeStyles(
                first="#10B981",    # Green for start
                last="#EF4444",     # Red for end  
                default="#3B82F6"   # Blue for regular nodes
            ),
            wrap_label_n_words=3,
            background_color="white",
            padding=10,
            draw_method=MermaidDrawMethod.API  # Uses mermaid.ink API
        )
        
        with open("rag_cag_styled.png", "wb") as f:
            f.write(png_data)
        
        print("🎨 Styled PNG saved to: rag_cag_styled.png")
        return True
        
    except ImportError:
        print("❌ Advanced styling requires additional packages")
        return False

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Generate all diagram formats"""
    print("🚀 RAG-CAG Workflow Diagram Generator")
    print("Using LangGraph built-in visualization methods")
    print("=" * 50)
    
    # Display graph structure
    display_graph_info()
    
    # Generate Mermaid code
    print("\n" + "=" * 50)
    generate_mermaid_code()
    
    # Generate PNG
    print("\n" + "=" * 50)
    if not generate_png_diagram():
        print("Mermaid code generated successfully!")
        print("   You can copy the .mmd file to mermaid.live to view")
    
    # Try advanced styling
    print("\n" + "=" * 50)
    generate_with_custom_styling()
    
    print("\n✅ Diagram generation complete!")
    print("\nFiles generated:")
    print("   📄 rag_cag_langgraph.mmd (Mermaid source)")
    print("   🖼️  rag_cag_workflow.png (if successful)")
    print("   🎨 rag_cag_styled.png (if successful)")

if __name__ == "__main__":
    main()