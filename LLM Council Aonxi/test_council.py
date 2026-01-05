#!/usr/bin/env python3
"""
Quick test of the LLM Council system
"""

from council_system import CouncilOrchestrator


def test_basic_functionality():
    """Test that the system runs without errors"""
    print("🧪 Testing LLM Council System...")

    # Initialize
    council = CouncilOrchestrator()

    # Test question
    test_question = "What are best practices for data privacy?"

    print(f"Question: {test_question}")
    print("Running deliberation...")

    try:
        decision = council.deliberate(test_question)

        print("\n✅ Test PASSED!")
        print(f"Answer generated: {len(decision.final_answer)} characters")
        print(f"Confidence: {decision.confidence:.1%}")
        print(f"Safety level: {decision.safety_level.value}")

        # Check audit log was created
        import os
        if os.path.exists("council_audit.log"):
            print(f"Audit log created: {os.path.getsize('council_audit.log')} bytes")

        return True

    except Exception as e:
        print(f"\n❌ Test FAILED: {e}")
        return False


if __name__ == "__main__":
    test_basic_functionality()
