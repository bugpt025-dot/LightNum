# LightNum

A native Luau arbitrary-magnitude number library for Roblox (`v1.9.1`).

LightNum provides full 53-bit IEEE-754 mantissa precision up to `10^(10^308)` and hyper-layered representation up to `10^^(2^1024)`. It is designed for incremental games, simulators, and high-throughput numeric workloads, while maintaining 100% wire and `OrderedDataStore` compatibility with GammaNum.

---

## Table of Contents

- [Architecture and Representation](#architecture-and-representation)
- [Installation](#installation)
- [Usage Modes](#usage-modes)
  - [1. Stateless Buffer API (Default)](#1-stateless-buffer-api-default)
  - [2. Table Wrapper API (`ln.wrap`)](#2-table-wrapper-api-lnwrap)
  - [3. Raw Register-Tuple API (`ln.Tuple`)](#3-raw-register-tuple-api-lntuple)
  - [4. Structure-of-Arrays Batch Engine (`ln.SoA`)](#4-structure-of-arrays-batch-engine-lnsoa)
  - [5. Extended Hyper Layer Engine (`ln.Hyper`)](#5-extended-hyper-layer-engine-lnhyper)
  - [6. Luau Native Vector Interop](#6-luau-native-vector-interop)
- [Economy and Game Math](#economy-and-game-math)
- [Serialization and DataStore Codecs](#serialization-and-datastore-codecs)
- [Formatting and Notation](#formatting-and-notation)
- [Complete API Reference](#complete-api-reference)

---

## Architecture and Representation

LightNum uses a canonical 3-component `(m, e, l)` model where `m` is the signed mantissa, `e` is the exponent, and `l` is the hyper-layer index:

| Layer (`l`) | Value Representation | Range | Precision |
| :--- | :--- | :--- | :--- |
| `0` | `m` | `(-10^10, -10^-10)`, `0`, `(10^-10, 10^10)` | Full 53-bit `f64` (~15–17 decimal digits) |
| `1` | `m * 10^e` | `abs(m)` in `[1, 10)`, `abs(e)` in `[10, 2^53]` | Full 53-bit `f64` (~15–17 decimal digits) |
| `2` | `sign(m) * 10^(±10^abs(e))` | Up to `10^(10^308)` | 53-bit exponent `e` |
| `>= 3` | `sign(m) * (10^^l)^abs(e)` | Up to `10^^(2^1024)` | Dominant-term arithmetic |

### Internal Binary Formats

- **25-Byte Native Buffer (Default):** Stores `sign` (`i8` at offset `0`), `l` (`f64` at offset `1`), `e` (`f64` at offset `9`), and `m` (`f64` at offset `17`). Avoids `log10` conversion overhead during arithmetic.
- **17-Byte Legacy Buffer (GammaNum Wire Format):** Stores `sign` (`i8` at offset `0`), `l17` (`f64` at offset `1`), and `e17` (`f64` at offset `9`). Fully interoperable with all LightNum functions and existing GammaNum saved data.

### Luau Compiler Optimizations

- Compiled with `--!native` and `--!optimize 2` for Luau Native Code Generation.
- Replaces dictionary hash lookups with 1-indexed array lookup tables (`negPow10`, `posPow10`, `pow10Arr`, `fullPow10`) in arithmetic hot paths.
- Uses specialized fast-path decoders (`fastUnpackPair`, `fastUnpackTriple`) for 25-byte buffers and Layer-0 scalar numbers.
- Minimal write-barrier mutators only write table slots (`[1]`, `[2]`, `[3]`) or buffer offsets when values actually change.

---

## Installation

### Option 1: Roblox Studio Model / Auto-Updating Package (Recommended)
1. Download `LightNum.rbxm` from the [Releases](../../releases) page.
2. Drag `LightNum.rbxm` into `ReplicatedStorage` inside Roblox Studio.
3. Because the asset is configured as a Roblox Package, you can pull future updates directly through Studio's Package Manager.

### Option 2: Wally
Add LightNum to your `wally.toml`:

```toml
[dependencies]
LightNum = "bugpt025-dot/lightnum@1.9.1"
```

### Option 3: Manual / Rojo
1. Create a `ModuleScript` named `LightNum` inside `ReplicatedStorage`.
2. Copy the contents of `init.luau` (or `LightNum.luau`) into the `ModuleScript`.

### Requiring the Module

```lua
local ReplicatedStorage = game:GetService("ReplicatedStorage")
local ln = require(ReplicatedStorage:WaitForChild("LightNum"))
```

---

## Usage Modes

All LightNum functions accept 25-byte buffers, 17-byte buffers, wrapped tables, plain `{m, e, l}` tables, Luau `number`s, and formatted `string`s interchangeably.

### 1. Stateless Buffer API (Default)

Standard constructors return a 25-byte `buffer`. For tight loops, use `addInto` or `mulInto` to write directly into a pre-allocated 25-byte buffer without heap allocations.

```lua
local cash = ln.fromNumber(1000)
local mult = ln.fromString("2.5e15")

cash = ln.mul(cash, mult)
print(ln.toSuffix(cash))     --> "2.50Qn"
print(ln.toScientific(cash)) --> "2.500e18"

-- Zero-allocation accumulation using a destination buffer
local accumulator = ln.new(0)
for i = 1, 1000 do
	ln.addInto(accumulator, accumulator, cash)
end
```

### 2. Table Wrapper API (`ln.wrap`)

`ln.wrap` (aliased as `ln.fast` and `ln.obj`) wraps `{m, e, l}` in a metatable supporting standard Luau operators (`+`, `-`, `*`, `/`, `//`, `%`, `^`, `-a`, `==`, `<`, `<=`) and in-place mutators (`:addeq`, `:subeq`, `:muleq`, `:diveq`, `:fmaeq`, `:poweq`).

```lua
local a = ln.wrap(1e50)
local b = ln.wrap("1e100")

-- Immutable operator chaining
local c = (a * b) ^ 2
print(c:toScientific()) --> "1.000e300"

-- In-place mutation (zero buffer allocation, minimal GC write barriers)
a:muleq(1.15)
a:addeq(5000)
a:fmaeq(b, 2) -- a = a + (b * 2)
```

### 3. Raw Register-Tuple API (`ln.Tuple`)

For maximum performance without allocating tables or buffers, operate directly on `(m, e, l)` stack values using `ln.Tuple` or top-level shorthand functions (`ln.addT`, `ln.mulT`, `ln.fmaT`, etc.).

```lua
local m, e, l = ln.Tuple.fromNumber(100)
local sm, se, sl = ln.Tuple.fromString("1.05e20")

for i = 1, 10000 do
	m, e, l = ln.mulT(m, e, l, sm, se, sl)
end

local resultBuffer = ln.pack(m, e, l)
```

### 4. Structure-of-Arrays Batch Engine (`ln.SoA`)

`LightNum.SoA` stores `N` numbers across three contiguous `f64` buffers (`m`, `e`, `l`). Ideal for updating thousands of currency balances, damage ticks, or multipliers per frame with zero garbage collection pressure.

```lua
local balances = ln.SoA.fromArray({ 100, 5000, "1e20", "1e100" })
local multipliers = ln.SoA.fromArray({ 2, 1.5, 10, 100 })

-- In-place vectorized operations
balances:mul(multipliers)
balances:mulScalar(1.05)
balances:addScalar(250)

-- Reductions
local total = balances:sum()
local product = balances:prod()
```

### 5. Extended Hyper Layer Engine (`ln.Hyper`)

For numbers exceeding standard `f64` layer indices (`l > 2^53`), `ln.Hyper` represents the layer counter as an arbitrary decimal string up to 400 digits while keeping IEEE-754 mantissa and exponent fields.

```lua
local h1 = ln.Hyper.fromParts(1, 500, "1000000000000000000000000")
local h2 = ln.Hyper.raiseLayer(h1, "500000")

local sum = h1 + h2
local saved = ln.Hyper.serialize(sum) -- "LNH1|1|500|1000000000000000000500000"
local loaded = ln.Hyper.deserialize(saved)
```

### 6. Luau Native Vector Interop

Opt-in conversion to Luau's native 3-component `vector` value type (`vector(m, e, l)`). Note that Luau vectors use 32-bit floats (~7 decimal digits), whereas core LightNum math remains 53-bit `f64`.

```lua
local vec = ln.toVector(cash)
local restored = ln.fromVector(vec)
```

---

## Economy and Game Math

LightNum includes closed-form geometric series solvers for incremental game upgrades, eliminating iterative loops when calculating bulk costs or maximum affordable purchases.

```lua
local currency = ln.fromString("1e25")
local baseCost = ln.fromNumber(100)
local growthRatio = ln.fromNumber(1.15)
local owned = ln.fromNumber(50)

-- Maximum number of upgrades affordable with current currency
local maxAffordable = ln.maxBuy(currency, baseCost, growthRatio, owned)

-- Total cost to purchase 25 upgrades starting from `owned`
local bulkCost = ln.costBulk(baseCost, growthRatio, owned, 25)

-- General geometric series: sum of base * mul^i for i = start..last
local seriesSum = ln.geosum(baseCost, growthRatio, 0, 99)

-- Inverse geometric series: finds `last` index given a target sum `amount`
local targetIndex = ln.geosumR(baseCost, growthRatio, 0, currency)
```

---

## Serialization and DataStore Codecs

| Format | Encode | Decode | Description |
| :--- | :--- | :--- | :--- |
| **Plain Table** | `ln.toTable(n)` | `ln.fromTable(tbl)` | Returns `{m, e, l}`. Directly compatible with `DataStoreService`, `RemoteEvent`, and JSON encoding. |
| **OrderedDataStore** | `ln.lbencode(n)` | `ln.lbdecode(score)` | Encodes into a monotonic `number` preserving strict ordering across positive and negative values for `OrderedDataStore` leaderboards (100% GammaNum compatible). |
| **Base64** | `ln.b64encode(n)` | `ln.b64decode(str)` | Encodes the 17-byte representation into a compact 24-character Base64 string. |
| **Hexadecimal** | `ln.hexencode(n)` | `ln.hexdecode(str)` | Encodes the 17-byte representation into a 34-character hex string. |
| **17-Byte Buffer** | `ln.tobuffer17(n)` | `ln.tobuffer(buf)` | Converts to legacy 17-byte GammaNum buffer format. |

```lua
-- OrderedDataStore Leaderboard Example
local score = ln.lbencode(cash)
OrderedDataStore:SetAsync(tostring(player.UserId), score)

-- Reading back from OrderedDataStore
local restoredCash = ln.lbdecode(score)
```

---

## Formatting and Notation

### Notation Functions

- `ln.tostring(n, suffixType?, intRound?)`: Formats according to `ln.DefaultSuffixType` (or an explicit `ln.SuffixTypes` enum).
- `ln.toSuffix(n)`: Standard incremental suffixes (`k`, `M`, `B`, `T`, `Qd`, `Qt`, `Sx`, `Sp`, `Oc`, `No`, up to `NoNi` and `Mi` at `10^3003`). Uses `1/` prefix for small values (`< 10^-3`).
- `ln.toScientific(n)`: Standard scientific notation (`1.234e56`, `e1.234e6`, or layered `1.234L+10` at high layers).
