# LightNum

A high-performance, native Luau big-number library for Roblox.

LightNum provides full 53-bit IEEE-754 precision up to $10^{(10^{308})}$ and hyper-layered representation up to $10\uparrow\uparrow(2^{1024})$. It is designed for incremental games, simulators, and high-throughput numeric workloads, while maintaining 100% wire and `OrderedDataStore` compatibility with GammaNum.

## Features

* **Precision and Range:** Maintains 15–17 significant decimal digits up to $10^{(10^{308})}$, with hyper-layered support up to $10\uparrow\uparrow(2^{1024})$. Includes an opt-in `LightNum.Hyper` engine supporting decimal-string layer indices up to 400 digits.
* **Native Luau Optimization:** Built with `--!native` and `--!optimize 2`, using 1-indexed array lookup tables, inlined Layer-0/1 fast paths, and specialized 25-byte buffer decoders.
* **Flexible Representation Modes:**
  * **Stateless Buffers (`25-byte` / `17-byte`):** Compact binary buffers with destination-buffer variants (`addInto`, `mulInto`) to avoid loop allocations.
  * **Table Wrappers (`ln.wrap`):** Standard Luau arithmetic operators (`+`, `-`, `*`, `/`, `^`, `<`, `==`) and minimal-write-barrier in-place mutators (`:addeq`, `:muleq`, `:fmaeq`).
  * **Register-Tuple Kernels (`ln.Tuple` / `ln.addT`):** Zero-allocation `(m, e, l)` tuple math for tight loops.
  * **Structure-of-Arrays (`LightNum.SoA`):** Batch operations across contiguous `f64` buffers without garbage collection overhead.
* **Serialization and Wire Compatibility:** Built-in support for plain JSON/DataStore tables (`{m, e, l}`), 17-byte GammaNum buffers, Base64, Hex, and monotonic `OrderedDataStore` encoding (`lbencode`/`lbdecode`).
* **Game Economy and Formatting Utilities:** Includes geometric series solvers (`costBulk`, `maxBuy`, `geosum`), interpolation (`lerp`, `logLerp`), and formatting for standard suffixes (`k` through `Mi`), Scientific, Engineering, Layered, and Hyper notations.

## Installation

1. Create a `ModuleScript` named `LightNum` in `ReplicatedStorage`.
2. Paste the contents of `LightNum.lua` into the module.
3. Require the module in your scripts:

```lua
local ReplicatedStorage = game:GetService("ReplicatedStorage")
local ln = require(ReplicatedStorage:WaitForChild("LightNum"))
```

## Usage

### Stateless Buffer API

Standard constructors return a 25-byte `buffer`. All functions accept buffers, numbers, strings, and wrapped tables interchangeably.

```lua
local cash = ln.fromNumber(1000)
local mult = ln.fromString("2.5e15")

cash = ln.mul(cash, mult)
print(ln.toSuffix(cash))     --> "2.50Qn"
print(ln.toScientific(cash)) --> "2.500e18"

-- Reuse a destination buffer in hot loops to avoid allocations
local accumulator = ln.new(0)
for i = 1, 1000 do
	ln.addInto(accumulator, accumulator, cash)
end
```

### Operator and Wrapper API (`ln.wrap`)

Use `ln.wrap()` for standard math operators and in-place mutation without allocating buffers.

```lua
local a = ln.wrap(1e50)
local b = ln.wrap("1e100")

local c = (a * b) ^ 2
print(c:toScientific()) --> "1.000e300"

-- In-place mutation
a:muleq(1.15)
a:addeq(5000)
```

### Economy Math

Geometric series helpers for upgrade pricing and bulk-purchasing:

```lua
local currency = ln.fromString("1e25")
local baseCost = ln.fromNumber(100)
local growthRatio = ln.fromNumber(1.15)
local owned = ln.fromNumber(50)

local affordable = ln.maxBuy(currency, baseCost, growthRatio, owned)
local bulkPrice = ln.costBulk(baseCost, growthRatio, owned, 25)
```

### DataStore and OrderedDataStore Serialization

```lua
-- Standard DataStore / RemoteEvent (plain table or Base64)
local serialized = ln.toTable(cash) -- { m, e, l }
local deserialized = ln.fromTable(serialized)

local b64 = ln.b64encode(cash)
local fromB64 = ln.b64decode(b64)

-- OrderedDataStore (17-byte GammaNum wire-compatible monotonic double)
local score = ln.lbencode(cash)
OrderedDataStore:SetAsync(player.UserId, score)

local restored = ln.lbdecode(score)
```

### Batch Processing (`LightNum.SoA`)

Process large arrays of numbers in contiguous `f64` buffers:

```lua
local balances = ln.SoA.fromArray({ 100, 5000, "1e20", "1e100" })
balances:mulScalar(1.05)
balances:addScalar(250)

local total = balances:sum()
print(ln.toSuffix(total))
```

## API Overview

| Category | Functions |
| :--- | :--- |
| **Constructors** | `new`, `fromNumber`, `fromString`, `fromScientific`, `fromTable`, `fromTuple` |
| **Arithmetic** | `add`, `sub`, `mul`, `div`, `fma`, `pow`, `root`, `sqrt`, `cbrt`, `log`, `log10`, `ln`, `exp`, `tetr`, `slog`, `gamma`, `fact` |
| **In-Place / Zero-Alloc** | `addInto`, `mulInto`, `addeq`, `subeq`, `muleq`, `diveq`, `fmaeq`, `poweq`, `ln.Tuple.*` (`addT`, `mulT`, etc.) |
| **Comparison** | `compare`, `eq`, `lt`, `lte`, `gt`, `gte`, `min`, `max`, `clamp` |
| **Economy and Math** | `geosum`, `geosumR`, `costBulk`, `maxBuy`, `lerp`, `invLerp`, `logLerp`, `hypot`, `random` |
| **Formatting** | `tostring`, `toSuffix`, `toScientific`, `toEngineer`, `toLayered`, `toHyper`, `suffix` |
| **Serialization** | `toTable`, `fromTable`, `tobuffer`, `tobuffer17`, `b64encode`/`decode`, `hexencode`/`decode`, `lbencode`/`decode` |
| **Batch / SIMD / Hyper** | `ln.SoA.*`, `ln.toVector`, `ln.fromVector`, `ln.Hyper.*` |

## Configuration

Default formatting behavior can be configured at runtime:

```lua
ln.DefaultSuffixType = ln.SuffixTypes.Scientific306 -- Switch to Scientific after 1e306
ln.DefaultDigits = 3                                -- Decimal digits in Scientific/Layered
ln.DefaultTotalDigits = 3                           -- Significant digits in Suffix/Engineer
ln.MaxEs = 3                                        -- Max 'e' chain before Layered notation (e.g., eee10)
```
