# Design Documents

This directory contains architectural analysis and design proposals for the faber-neuroevolution system.

## Documents

### [ARCHITECTURE_ANALYSIS.md](./ARCHITECTURE_ANALYSIS.md)

Analysis of the current architecture, identifying:
- The "bridge problem" - Erlang code living in Elixir project
- Why Elixir exists (only for Phoenix UI)
- Component responsibilities and boundaries
- Short-term fix: move bridge to faber-neuroevolution

### [EVENT_DRIVEN_ARCHITECTURE.md](./EVENT_DRIVEN_ARCHITECTURE.md)

Proposal for event-driven communication:
- Replace direct calls with message passing
- Replace callbacks with PubSub topics
- Enable distributed evaluation via Macula mesh
- Event types and topic design
- Implementation phases

---

## Key Insight

**The bridge problem disappears with event-driven architecture.**

Current:
```
neuroevolution_server → calls → evaluator_bridge → calls → Elixir evaluator
```

Proposed:
```
neuroevolution_server → publishes → {evaluate_request}
                                          ↓
                              (any subscriber, any language)
                                          ↓
                        evaluator → publishes → {evaluated}
```

No bridge needed - just agreement on message format.

---

## Implementation Priority

1. **Immediate:** Move `neurolab_evaluator_bridge.erl` to faber-neuroevolution
2. **Next:** Add `neuroevolution_events` abstraction
3. **Then:** Create evaluator worker with event-based communication
4. **Finally:** Integrate with Macula mesh PubSub
