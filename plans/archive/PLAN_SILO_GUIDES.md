# PLAN: Silo Guides with SVG Diagrams

**Status:** Complete
**Created:** 2025-12-23
**Last Updated:** 2025-12-23
**Related:** PLAN_*_SILO.md (all silo plans), liquid-conglomerate.md

---

## Overview

Create comprehensive educational guides for all 13 Liquid Conglomerate silos. Each guide will provide extensive documentation on how the silo works, how it integrates with the neuroevolution engine, and include professional SVG diagrams.

## Directory Structure

```
guides/
├── silos/
│   ├── assets/
│   │   ├── task-silo-architecture.svg
│   │   ├── task-silo-dynamics.svg
│   │   ├── task-silo-dataflow.svg
│   │   ├── resource-silo-architecture.svg
│   │   ├── resource-silo-dynamics.svg
│   │   ├── resource-silo-dataflow.svg
│   │   ├── distribution-silo-architecture.svg
│   │   ├── distribution-silo-dynamics.svg
│   │   ├── distribution-silo-dataflow.svg
│   │   ├── ... (3 SVGs per silo - architecture, dynamics, dataflow)
│   │   ├── lc-supervisor-architecture.svg    # LC Management
│   │   ├── lc-cross-silo-flow.svg            # LC Management
│   │   └── lc-hierarchical-control.svg       # LC Management
│   ├── lc-overview.md              # LC management/supervision/config guide
│   ├── task-silo.md
│   ├── resource-silo.md
│   ├── distribution-silo.md
│   ├── temporal-silo.md
│   ├── competitive-silo.md
│   ├── developmental-silo.md
│   ├── regulatory-silo.md
│   ├── economic-silo.md
│   ├── communication-silo.md
│   ├── social-silo.md
│   ├── cultural-silo.md
│   ├── ecological-silo.md
│   ├── morphological-silo.md
│   └── index.md                    # Silo overview/navigation
└── liquid-conglomerate.md          # Updated with velocity table
```

---

## Guide Template Structure

Each silo guide follows this structure:

```markdown
# [Silo Name] Silo Guide

## What is the [Silo Name] Silo?
- High-level purpose (2-3 paragraphs, educational)
- Real-world analogies to explain concepts
- Why this silo matters for neuroevolution

## Architecture Overview
![Silo Architecture](assets/[silo]-architecture.svg)
- Component diagram
- Internal subsystems explanation

## How It Works

### Sensors (Inputs)
- What the silo observes
- How sensors are collected
- Normalization and ranges

### Actuators (Outputs)
- What the silo controls
- How actuator values affect evolution
- Default values and ranges

### The Control Loop
![Control Loop](assets/[silo]-control-loop.svg)
- Sensor → TWEANN → Actuator flow
- Update frequency
- L2 guidance integration

## Integration with the Neuroevolution Engine

### Wiring Diagram
![Dataflow](assets/[silo]-dataflow.svg)
- Where data comes from
- Where outputs go
- Event emissions

### Cross-Silo Interactions
![Cross-Silo](assets/[silo]-cross-silo.svg)
- Signals sent to other silos
- Signals received from other silos
- Coordination patterns

### Engine Integration Points
- Which engine modules interact with this silo
- Callbacks and hooks
- Configuration options

## Training Velocity Impact
- Performance table (already in PLAN files)
- When to enable/disable
- Trade-offs

## Practical Examples

### Example 1: [Scenario]
```erlang
%% Code example
```

### Example 2: [Scenario]
```erlang
%% Code example
```

## Tuning Guide
- Key parameters to adjust
- Common pitfalls
- Debugging tips

## Events Reference
- All events emitted
- Event payloads
- Storage and replay

## Further Reading
- Related guides
- Academic references
- Source code locations
```

---

## SVG Diagram Requirements

### Per-Silo Diagrams (2-4 each)

| Diagram Type | Purpose | Style |
|--------------|---------|-------|
| **Architecture** | Show internal components | Boxes, hierarchy |
| **Control Loop** | Sensor → TWEANN → Actuator | Flowchart, circular |
| **Dataflow** | Integration with engine | Arrows, swim lanes |
| **Cross-Silo** | Interactions with other silos | Network diagram |

### Global Diagrams (shared)

| Diagram | Purpose | Location |
|---------|---------|----------|
| **13-Silo Matrix** | All silo interactions | assets/silo-interaction-matrix.svg |
| **LC Full Architecture** | Complete LC overview | assets/lc-full-architecture.svg |
| **Hierarchy Levels** | L0/L1/L2 visualization | assets/lc-hierarchy.svg |

### SVG Design Standards

```
Colors:
- Task Silo:          #4A90D9 (blue)
- Resource Silo:      #7B68EE (purple)
- Distribution Silo:  #20B2AA (teal)
- Temporal Silo:      #FFB347 (orange)
- Competitive Silo:   #DC143C (crimson)
- Developmental Silo: #32CD32 (lime)
- Regulatory Silo:    #9370DB (medium purple)
- Economic Silo:      #FFD700 (gold)
- Communication Silo: #00CED1 (dark turquoise)
- Social Silo:        #FF69B4 (hot pink)
- Cultural Silo:      #8B4513 (saddle brown)
- Ecological Silo:    #228B22 (forest green)
- Morphological Silo: #708090 (slate gray)

Typography:
- Headers: 16px bold
- Labels: 12px regular
- Annotations: 10px italic

Styling:
- Rounded corners (rx=5)
- Drop shadows for depth
- Gradient fills for emphasis
- Consistent arrow styles
```

---

## Files to Create

### Phase 1: Core Silos (existing in engine)

| File | SVG Diagrams | Priority |
|------|--------------|----------|
| `silos/index.md` | 1 (overview) | P0 |
| `silos/task-silo.md` | 3 | P0 |
| `silos/resource-silo.md` | 3 | P0 |
| `silos/distribution-silo.md` | 3 | P0 |

### Phase 2: Performance Silos

| File | SVG Diagrams | Priority |
|------|--------------|----------|
| `silos/temporal-silo.md` | 4 | P1 |
| `silos/economic-silo.md` | 3 | P1 |
| `silos/morphological-silo.md` | 3 | P1 |

### Phase 3: Evolutionary Silos

| File | SVG Diagrams | Priority |
|------|--------------|----------|
| `silos/competitive-silo.md` | 4 | P2 |
| `silos/cultural-silo.md` | 3 | P2 |
| `silos/social-silo.md` | 3 | P2 |
| `silos/ecological-silo.md` | 4 | P2 |

### Phase 4: Advanced Silos

| File | SVG Diagrams | Priority |
|------|--------------|----------|
| `silos/developmental-silo.md` | 3 | P3 |
| `silos/regulatory-silo.md` | 3 | P3 |
| `silos/communication-silo.md` | 4 | P3 |

### Total SVG Requirements

- **Per-silo diagrams**: ~42 SVGs (13 silos × ~3.2 avg)
- **Global diagrams**: 3 SVGs
- **Total**: ~45 SVG diagrams

---

## Guide Content Sources

Each guide draws from:

1. **PLAN files** (`plans/PLAN_*_SILO.md`)
   - Sensors and actuators specifications
   - Record definitions
   - Event payloads
   - Cross-silo signal matrix

2. **Existing source code** (when implemented)
   - Actual module names
   - API documentation
   - Configuration options

3. **Academic references**
   - Referenced in PLAN files
   - Additional context where needed

4. **Analogies and examples**
   - Real-world parallels
   - Practical use cases

---

## Silo Guide Content Specifications

### 1. Task Silo Guide

**Educational Focus:**
- What is "task" in neuroevolution context
- Fitness evaluation concepts
- Stagnation detection and recovery

**Key Topics:**
- Fitness landscape visualization
- Multi-objective optimization
- Generational vs steady-state
- Stagnation detection mechanisms

**Analogies:**
- Task silo as "coach" - observes performance, adjusts training
- Fitness landscape as "terrain" - hills, valleys, plateaus

**SVG Diagrams:**
1. `task-silo-architecture.svg` - Internal components
2. `task-silo-fitness-landscape.svg` - Fitness concepts
3. `task-silo-stagnation.svg` - Detection and recovery

---

### 2. Resource Silo Guide

**Educational Focus:**
- Computational budget management
- Preventing resource waste
- Dynamic allocation strategies

**Key Topics:**
- Evaluation time budgeting
- Memory management
- Compute throttling
- Priority scheduling

**Analogies:**
- Resource silo as "CFO" - manages budget
- Compute as "currency" - spend wisely

**SVG Diagrams:**
1. `resource-silo-architecture.svg` - Internal components
2. `resource-silo-allocation.svg` - Resource distribution
3. `resource-silo-throttling.svg` - Dynamic adjustment

---

### 3. Distribution Silo Guide

**Educational Focus:**
- Population structure concepts
- Island models and migration
- Diversity maintenance

**Key Topics:**
- Population topology
- Migration patterns
- Species formation
- Geographic isolation

**Analogies:**
- Distribution as "geography" - islands, continents
- Migration as "trade routes" - genetic exchange

**SVG Diagrams:**
1. `distribution-silo-architecture.svg` - Internal components
2. `distribution-silo-topology.svg` - Population structures
3. `distribution-silo-migration.svg` - Flow patterns

---

### 4. Temporal Silo Guide

**Educational Focus:**
- Time management in evolution
- Episode duration optimization
- Early termination benefits

**Key Topics:**
- Evaluation timeouts
- Episode length adaptation
- Patience and early stopping
- Credit assignment horizons

**Analogies:**
- Temporal silo as "timekeeper" - knows when to stop
- Episodes as "experiments" - some need more time

**SVG Diagrams:**
1. `temporal-silo-architecture.svg` - Internal components
2. `temporal-silo-timeline.svg` - Time management
3. `temporal-silo-early-term.svg` - Early termination logic
4. `temporal-silo-convergence.svg` - Convergence detection

---

### 5. Competitive Silo Guide

**Educational Focus:**
- Adversarial dynamics
- Elo rating systems
- Arms race prevention

**Key Topics:**
- Opponent archives
- Matchmaking algorithms
- Skill rating (Elo)
- Counter-strategy detection

**Analogies:**
- Competitive silo as "league manager" - organizes matches
- Elo as "skill ladder" - fair matchups

**SVG Diagrams:**
1. `competitive-silo-architecture.svg` - Internal components
2. `competitive-silo-matchmaking.svg` - Pairing logic
3. `competitive-silo-archive.svg` - Opponent storage
4. `competitive-silo-arms-race.svg` - Cycle detection

---

### 6. Cultural Silo Guide

**Educational Focus:**
- Cultural vs genetic evolution
- Knowledge transfer mechanisms
- Cumulative culture effects

**Key Topics:**
- Innovation discovery
- Tradition maintenance
- Imitation learning
- Cultural ratchet

**Analogies:**
- Cultural silo as "library" - stores and spreads knowledge
- Traditions as "best practices" - proven solutions

**SVG Diagrams:**
1. `cultural-silo-architecture.svg` - Internal components
2. `cultural-silo-transmission.svg` - Knowledge flow
3. `cultural-silo-ratchet.svg` - Cumulative effect

---

### 7. Social Silo Guide

**Educational Focus:**
- Social structure in populations
- Reputation and trust
- Cooperation evolution

**Key Topics:**
- Reputation tracking
- Coalition formation
- Kin selection
- Reciprocal altruism

**Analogies:**
- Social silo as "community manager" - tracks relationships
- Reputation as "credit score" - affects opportunities

**SVG Diagrams:**
1. `social-silo-architecture.svg` - Internal components
2. `social-silo-network.svg` - Social connections
3. `social-silo-cooperation.svg` - Cooperation dynamics

---

### 8. Ecological Silo Guide

**Educational Focus:**
- Environmental pressures
- Resource dynamics
- Catastrophic events

**Key Topics:**
- Carrying capacity
- Disease modeling
- Environmental cycles
- Mass extinction/recovery

**Analogies:**
- Ecological silo as "weather system" - creates pressure
- Resources as "food supply" - competition driver

**SVG Diagrams:**
1. `ecological-silo-architecture.svg` - Internal components
2. `ecological-silo-environment.svg` - Pressure sources
3. `ecological-silo-cycles.svg` - Seasonal patterns
4. `ecological-silo-catastrophe.svg` - Extinction events

---

### 9. Morphological Silo Guide

**Educational Focus:**
- Network structure control
- Size vs efficiency trade-offs
- Pruning and growth

**Key Topics:**
- Neuron/connection budgets
- Complexity penalties
- Automatic pruning
- Hardware targeting

**Analogies:**
- Morphological silo as "architect" - controls structure
- Network size as "building footprint" - constraints matter

**SVG Diagrams:**
1. `morphological-silo-architecture.svg` - Internal components
2. `morphological-silo-network.svg` - Structure visualization
3. `morphological-silo-pruning.svg` - Simplification process

---

### 10. Developmental Silo Guide

**Educational Focus:**
- Lifetime development
- Critical periods
- Plasticity management

**Key Topics:**
- Developmental stages
- Critical period timing
- Plasticity decay
- Metamorphosis

**Analogies:**
- Developmental silo as "lifecycle manager" - guides growth
- Critical periods as "learning windows" - timing matters

**SVG Diagrams:**
1. `developmental-silo-architecture.svg` - Internal components
2. `developmental-silo-stages.svg` - Development timeline
3. `developmental-silo-plasticity.svg` - Plasticity curve

---

### 11. Regulatory Silo Guide

**Educational Focus:**
- Gene expression control
- Context-dependent behavior
- Epigenetic inheritance

**Key Topics:**
- Module activation
- Context switching
- Dormant capabilities
- Expression thresholds

**Analogies:**
- Regulatory silo as "gene switch" - controls what's active
- Dormant genes as "hidden features" - await activation

**SVG Diagrams:**
1. `regulatory-silo-architecture.svg` - Internal components
2. `regulatory-silo-expression.svg` - Gene activation
3. `regulatory-silo-context.svg` - Context switching

---

### 12. Economic Silo Guide

**Educational Focus:**
- Resource economics
- Budget optimization
- Trade and exchange

**Key Topics:**
- Compute budgeting
- Energy management
- Wealth redistribution
- Investment strategies

**Analogies:**
- Economic silo as "central bank" - manages economy
- Compute budget as "currency" - earn and spend

**SVG Diagrams:**
1. `economic-silo-architecture.svg` - Internal components
2. `economic-silo-budget.svg` - Resource allocation
3. `economic-silo-trade.svg` - Exchange patterns

---

### 13. Communication Silo Guide

**Educational Focus:**
- Language evolution
- Signal honesty
- Coordination mechanisms

**Key Topics:**
- Vocabulary evolution
- Deception detection
- Coordination protocols
- Dialect formation

**Analogies:**
- Communication silo as "translator" - manages signals
- Vocabulary as "shared language" - evolves over time

**SVG Diagrams:**
1. `communication-silo-architecture.svg` - Internal components
2. `communication-silo-vocabulary.svg` - Signal evolution
3. `communication-silo-coordination.svg` - Team communication
4. `communication-silo-dialects.svg` - Language divergence

---

## Success Criteria

### Per-Guide Requirements

- [ ] 1500-3000 words of educational content
- [ ] 2-4 professional SVG diagrams
- [ ] At least 2 practical code examples
- [ ] Cross-references to related guides
- [ ] Academic references where applicable
- [ ] Tuning recommendations

### Overall Requirements

- [x] All 13 silo guides complete
- [x] LC Overview guide (management/supervision/configuration)
- [ ] Index page with navigation
- [x] Consistent styling across all guides
- [x] All SVG diagrams follow design standards (3 per silo + 3 for LC)
- [ ] Links verified in liquid-conglomerate.md
- [x] Mobile-friendly SVG scaling

---

## Implementation Phases

### Phase 0: Setup (COMPLETE)
- [x] Create `guides/silos/` directory
- [x] Create `guides/silos/assets/` directory
- [x] Create index.md template
- [x] Establish SVG template

### Phase 1: Core Silos (COMPLETE)
- [x] Task Silo guide + SVGs (architecture, dynamics, dataflow)
- [x] Resource Silo guide + SVGs (architecture, dynamics, dataflow)
- [x] Distribution Silo guide + SVGs (architecture, dynamics, dataflow)

### Phase 2: Performance Silos (COMPLETE)
- [x] Temporal Silo guide + SVGs (architecture, dynamics, dataflow)
- [x] Economic Silo guide + SVGs (architecture, dynamics, dataflow)
- [x] Morphological Silo guide + SVGs (architecture, dynamics, dataflow)

### Phase 3: Interaction Silos (COMPLETE)
- [x] Competitive Silo guide + SVGs (architecture, dynamics, dataflow)
- [x] Cultural Silo guide + SVGs (architecture, dynamics, dataflow)
- [x] Social Silo guide + SVGs (architecture, dynamics, dataflow)
- [x] Ecological Silo guide + SVGs (architecture, dynamics, dataflow)

### Phase 4: Advanced Silos (COMPLETE)
- [x] Developmental Silo guide + SVGs (architecture, dynamics, dataflow)
- [x] Regulatory Silo guide + SVGs (architecture, dynamics, dataflow)
- [x] Communication Silo guide + SVGs (architecture, dynamics, dataflow)

### Phase 5: LC Management (COMPLETE)
- [x] LC Overview guide (lc-overview.md)
- [x] Supervisor architecture SVG (lc-supervisor-architecture.svg)
- [x] Cross-silo flow SVG (lc-cross-silo-flow.svg)
- [x] Hierarchical control SVG (lc-hierarchical-control.svg)

### Phase 6: Polish (TODO)
- [ ] Global diagrams (13-silo matrix)
- [ ] Cross-link verification
- [ ] Style consistency check
- [ ] Index page finalization

---

## Notes

- SVG diagrams should be created using consistent tooling (Inkscape, Figma, or code-generated)
- All diagrams must be accessibility-friendly (text labels, not images of text)
- Consider dark mode compatibility for SVGs
- Diagrams should scale well from mobile to desktop
