#!/usr/bin/env python3
"""
LLM Council - Simple Runner
Usage: python run.py "Your question here"
"""

import sys
import json
from council_system import CouncilOrchestrator, asdict

def run_single_question(question):
    """Run council deliberation on a single question"""

    print("\n" + "═" * 70)
    print("🤖 LLM COUNCIL - AONXI DECISION ENGINE")
    print("═" * 70)

    # Initialize the council
    council = CouncilOrchestrator()

    # Run deliberation
    print(f"\n📋 Question: {question}")
    print("\n" + "─" * 70)

    decision = council.deliberate(question)

    # Display results
    print("\n" + "═" * 70)
    print("✅ FINAL DECISION")
    print("═" * 70)

    print(f"\n📝 Answer: {decision.final_answer}")
    print(f"\n📊 Confidence: {decision.confidence:.1%}")
    print(f"🏆 Winning Agent: {decision.winning_agent_id}")
    print(f"🚦 Safety Level: {decision.safety_level.value}")

    if decision.risks:
        print(f"\n⚠️  Risks Identified:")
        for risk in decision.risks:
            print(f"   • {risk}")

    if decision.citations:
        print(f"\n📚 Citations ({len(decision.citations)}):")
        for citation in decision.citations[:3]:  # Show first 3
            print(f"   • {citation}")
        if len(decision.citations) > 3:
            print(f"   • ... and {len(decision.citations) - 3} more")

    print(f"\n🔗 Audit Hash: {decision.audit_hash}")
    print(f"🕒 Timestamp: {decision.timestamp}")

    # Save to file
    import hashlib
    safe_name = hashlib.md5(question.encode()).hexdigest()[:8]
    filename = f"decision_{safe_name}.json"

    with open(filename, 'w') as f:
        json.dump(asdict(decision), f, indent=2)

    print(f"\n💾 Decision saved to: {filename}")
    print("═" * 70)

    return decision

def run_demo_mode():
    """Run with example questions"""
    questions = [
        "What are the most effective risk mitigation strategies for deploying AI in financial services?",
        "How should a startup prioritize between rapid growth and long-term stability?",
        "What factors should a company consider when expanding to international markets?",
        "How can organizations balance innovation with regulatory compliance?"
    ]

    print("\n" + "═" * 70)
    print("🎮 DEMO MODE - Running 4 example questions")
    print("═" * 70)

    for i, question in enumerate(questions, 1):
        print(f"\n\n{'='*70}")
        print(f"QUESTION {i}/4")
        print(f"{'='*70}")
        run_single_question(question)

    print("\n" + "═" * 70)
    print("✨ DEMO COMPLETE")
    print("Check the audit/ folder for logs and *.json files for decisions")
    print("═" * 70)

def main():
    """Main entry point"""

    # Check if question was provided as argument
    if len(sys.argv) > 1:
        # Join all arguments as the question
        question = " ".join(sys.argv[1:])
        run_single_question(question)
    else:
        # No arguments, show help
        print("\n" + "═" * 70)
        print("LLM Council - Decision Engine")
        print("═" * 70)
        print("\nUsage:")
        print("  python run.py \"Your question here\"")
        print("\nExamples:")
        print("  python run.py \"How should we approach AI safety?\"")
        print("  python run.py demo    # Run demo questions")
        print("\nOptions:")
        print("  --help              Show this help")
        print("  demo                Run demonstration mode")
        print("  \"question\"         Process a specific question")
        print("═" * 70)

        # Ask user if they want to run demo
        response = input("\nWould you like to run the demo? (y/n): ").lower()
        if response in ['y', 'yes']:
            run_demo_mode()
        else:
            print("\nPlease provide a question. Example:")
            print('python run.py "What is your advice for startups?"')

if __name__ == "__main__":
    main()