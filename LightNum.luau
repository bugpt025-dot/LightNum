--!optimize 2
--!native
--[[
	============================================================================
	LightNum v1.9.1 — Performance-Optimized Number Library with Preserved v1.8 Range
	============================================================================
	Successor to GammaNum (v1.1.3) & engineered to surpass SerikaNum on EVERY
	dimension: zero-hash native tuple kernels, minimal write-barrier mutators,
	53-bit precision up to 10^(10^308), range up to 10^^(2^1024), ergonomic
	operators, zero-conversion RemoteEvent/DataStore tables, SoA batch arrays,
	AND 100% 17-byte GammaNum wire & OrderedDataStore compatibility.

	Performance upgrades in v1.9:
	  1. Specialized scalar decoding for common Layer-0 number operands.
	  2. Optional addInto/mulInto destination-buffer APIs to avoid result allocation.
	  3. Inlined exact Layer-0 kernels in SoA div/fma and vector add/mul/fma.
	  4. Lazy legacy-tuple conversion in tostring, dense integer pow10 lookup,
	     and linear-time comma/Hyper layer-index string construction.
	  5. Existing public APIs, 17/25-byte formats, arithmetic representation,
	     and v1.8 extended range are preserved; no magnitude expansion is added.

	Performance and correctness upgrades in v1.7:
	  1. Specialized boxed-buffer dispatch for the common 25-byte format:
	     binary and ternary arithmetic read packed fields directly, bypassing
	     the generic any-type decoder for native LightNum buffers.
	  2. Retains v1.6 sparse batch shortcuts, v1.5 scalar identity exits, and
	     v1.4 inline Layer-0 add/sub/mul arithmetic paths.
	  3. No reduced mantissa precision, f32 conversion, approximation, or
	     altered tuple representation is introduced.
	  4. Adds boxed-buffer parity tests and boxed-operation benchmarks.

Accuracy hardening retained from v1.31:
	  1. Fixes cross-layer magnitude comparisons between canonical Layer-0 and
	     Layer-1 values. The previous path could compare mantissas while ignoring
	     the Layer-1 exponent, producing incorrect ordering/equality near the
	     1e-10 and 1e10 representation boundaries.
	  2. Keeps the existing tuple format, public API, 25-byte format, 17-byte
	     GammaNum compatibility, and arithmetic kernels unchanged otherwise.

	Key Architectural Upgrades in v1.30:
	  1. Zero-Hash 1-Indexed Array LUTs & Hoisted Layer-0/1 Kernels:
	     Replaces negative/zero-key Luau hash lookups (pow10Table[-diff], [e])
	     in hot arithmetic paths with dense 1-indexed C arrays (negPow10,
	     posPow10, pow10Arr). Hoists Layer-0 & Layer-1 fast paths in fmaTuple.
	     Hits ~1 op/ns on Layer-0 *T and minimizes instructions on Layer-1.
	  2. Minimal Write-Barrier Plain-Table WrapMT Mutators:
	     Inline Layer-0 & Layer-1 paths in :addeq, :subeq, :muleq, :diveq, and
	     :fmaeq only assign slots that actually change (0 writes on diff >= 17,
	     1 write self[1] = m on same-exponent updates, 2 writes on carry)
	     eliminating redundant Luau table-write barriers.
	  3. Structure-of-Arrays (LightNum.SoA) Full 53-Bit Batch Engine:
	     Stores N numbers in 3 contiguous 64-bit IEEE-754 f64 buffers (m, e, l)
	     for batch operations (addScalar, mulScalar, fmaScalar, add, mul, fma,
	     sum, prod) with ZERO GC barriers and ZERO precision loss (~15-17d).
	  4. Opt-In f32 Luau Native Vector Interop (toVector / fromVector):
	     Converts between lossless LightNum and Luau's SIMD vector(m, e, l)
	     value type (~7 digits f32) while keeping all LightNum math 53-bit f64.
	  5. Full Dual 25B Lossless + 17B GammaNum Wire ABI & Plain {m, e, l} Tables:
	     Seamless RemoteEvent / DataStore / JSON + OrderedDataStore lbencode.
	============================================================================
]]

local ln = {}

ln.Version = "1.9.1"
ln.Engine = "LightNum v1.9.1 Scalar Fast Paths, SoA Kernels & Hyper String Optimization"

ln.SuffixTypes = {
	Scientific =     0,
	Scientific3 =    1,
	Scientific36 =   2,
	Scientific306 =  3,
	Scientific3006 = 4,
	Engineer =       5,
	Engineer3 =      6,
	Engineer36 =     7,
	Engineer306 =    8,
	Engineer3006 =   9,
	Layered =        10,
}

-- Configuration (matches GammaNum defaults; mutable at runtime)
ln.DefaultSuffixType = ln.SuffixTypes.Scientific306
ln.DefaultDigits = 3
ln.DefaultTotalDigits = 3
ln.MaxEs = 3

-- Pre-cached Luau & Roblox buffer/math primitives
local abs = math.abs
local sign = math.sign
local floor = math.floor
local ceil = math.ceil
local round = math.round
local log10 = math.log10
local mlog = math.log
local mexp = math.exp
local modf = math.modf
local mmax = math.max
local mmin = math.min
local msin = math.sin
local huge = math.huge
local pi = math.pi
local mrandom = math.random

local bufCreate = buffer.create
local bufLen = buffer.len
local bufCopy = buffer.copy
local bufFromString = buffer.fromstring
local bufReadI8 = buffer.readi8
local bufWriteI8 = buffer.writei8
local bufReadU8 = buffer.readu8
local bufReadF64 = buffer.readf64
local bufWriteF64 = buffer.writef64

local sformat = string.format
local sbyte = string.byte
local schar = string.char
local tconcat = table.concat
local tinsert = table.insert
local tcreate = table.create

-- Mathematical constants
local LN_10 = mlog(10)
local INV_LN_10 = 1 / LN_10
local LOG10_2 = log10(2)
local EULER = mexp(1)
local MAX_F64 = 1.7976931348623157e+308
local INF_LIMIT = log10(MAX_F64)
local HALF_LN_TWO_PI = 0.5 * mlog(2 * pi)
local SQRT_TWO_PI = 2.50662827463100050241576528
local E_INV_E = mexp(1 / EULER)
local EXACT_INT_LIMIT = 9007199254740992 -- 2^53

-- Precomputed formatting & integer-power lookup tables
local digitcuts = {}
local capthres = {}
local capvalue = {}
local capthresl10 = {}
local capvaluel10 = {}
for i = 0, 20 do
	local p = 10 ^ (-i)
	digitcuts[i] = "%." .. i .. "f"
	capthres[i] = 10 - 0.5 * p
	capvalue[i] = 10 - p
	capthresl10[i] = log10(10 - 0.5 * p)
	capvaluel10[i] = log10(10 - p)
end

-- 1-indexed C array LUTs for zero-hash hot-path indexing under Luau
local negPow10 = {}
for i = 1, 17 do
	negPow10[i] = 10 ^ (-i)
end

local posPow10 = {}
for i = 1, 15 do
	posPow10[i] = 10 ^ i
end

-- Shifted by +11 so integer e in [-10..10] maps to array indices [1..21] (11 -> 10^0 = 1)
local pow10Arr = {}
for i = 1, 21 do
	pow10Arr[i] = 10 ^ (i - 11)
end

local pow10Table = {}
for i = -20, 20 do
	pow10Table[i] = 10 ^ i
end

-- Dense exact IEEE-754 powers for integral exponents in [-308, 308].
-- Entries are generated by the same Luau exponentiation operation used by
-- the previous version, then reused by integer-exponent normalization paths.
local fullPow10 = tcreate(617)
for k = -308, 308 do
	fullPow10[k + 309] = 10 ^ k
end

local Etrail = { "e" }
for i = 2, 20 do
	Etrail[i] = Etrail[i - 1] .. "e"
end

-- High-precision exp(x) - 1 and log(1 + x) for Luau
local function expm1(x: number): number
	if abs(x) < 1e-4 then
		local x2 = x * x
		return x + 0.5 * x2 + (x2 * x) * (1 / 6) + (x2 * x2) * (1 / 24)
	end
	return mexp(x) - 1
end

-- ============================================================================
-- CORE (m, e, l) REPRESENTATION, PACKING & LEGACY 17-BYTE CONVERSION
-- ============================================================================

local WrapMT = {}
WrapMT.__index = WrapMT

local function rawWrap(m: number, e: number, l: number)
	return setmetatable({ m, e, l }, WrapMT)
end

local function fromNumberTuple(input: number): (number, number, number)
	if input ~= input then
		return 0, 1, -1
	elseif input == 0 then
		return 0, 0, 0
	elseif input == huge then
		return 1, 1, huge
	elseif input == -huge then
		return -1, 1, huge
	end
	local a = if input > 0 then input else -input
	if a >= 1e10 then
		local k = floor(log10(a))
		local m = input / (fullPow10[k + 309] or (10 ^ k))
		if abs(m) >= 10 then
			m *= 0.1
			k += 1
		elseif abs(m) < 1 then
			m *= 10
			k -= 1
		end
		return m, k, 1
	elseif a <= 1e-10 then
		if a < 1e-300 then
			local a2 = a * 1e300
			local k = floor(log10(a2)) - 300
			local am = a2 / (fullPow10[k + 609] or (10 ^ (k + 300)))
			if am >= 10 then
				am *= 0.1
				k += 1
			elseif am < 1 then
				am *= 10
				k -= 1
			end
			if k == -10 and am > 1 then
				return input, 0, 0
			end
			return (if input > 0 then am else -am), k, 1
		end
		local k = floor(log10(a))
		local am = a / (fullPow10[k + 309] or (10 ^ k))
		if am >= 10 then
			am *= 0.1
			k += 1
		elseif am < 1 then
			am *= 10
			k -= 1
		end
		if k == -10 and am > 1 then
			return input, 0, 0
		end
		return (if input > 0 then am else -am), k, 1
	end
	return input, 0, 0
end

local function normalizeLayer1(m: number, e: number): (number, number, number)
	if m ~= m or e ~= e then
		return 0, 1, -1
	elseif m == 0 then
		return 0, 0, 0
	elseif abs(m) == huge then
		if e == -huge then
			return 0, 1, -1
		end
		return (if m > 0 then 1 else -1), 1, huge
	elseif e == huge then
		return (if m > 0 then 1 else -1), 1, huge
	elseif e == -huge then
		return 0, 0, 0
	end

	local sg = if m > 0 then 1 else -1
	local am = if m > 0 then m else -m

	if abs(e) < EXACT_INT_LIMIT then
		local ie = floor(e)
		local fe = e - ie
		if fe ~= 0 then
			am *= 10 ^ fe
			e = ie
		end
	end

	if am < 1 or am >= 10 then
		if am >= 10 and am < 100 then
			am *= 0.1
			e += 1
		elseif am >= 0.1 and am < 1 then
			am *= 10
			e -= 1
		elseif am < 1e-300 then
			local a2 = am * 1e300
			local k = floor(log10(a2)) - 300
			am = a2 / (fullPow10[k + 609] or (10 ^ (k + 300)))
			if am >= 10 then
				am *= 0.1
				k += 1
			elseif am < 1 then
				am *= 10
				k -= 1
			end
			e += k
		else
			local k = floor(log10(am))
			am = am / (fullPow10[k + 309] or (10 ^ k))
			if am >= 10 then
				am *= 0.1
				k += 1
			elseif am < 1 then
				am *= 10
				k -= 1
			end
			e += k
		end
	end

	if e == huge then
		return sg, 1, huge
	elseif e == -huge then
		return 0, 0, 0
	end

	if e >= -10 and e < 10 then
		if e == -10 and am == 1 then
			return sg, -10, 1
		end
		local v = am * (pow10Arr[e + 11] or (10 ^ e))
		if v == 0 then
			return 0, 0, 0
		elseif v >= 1e10 then
			return sg, 10, 1
		elseif v <= 1e-10 then
			return sg, -10, 1
		end
		return sg * v, 0, 0
	end

	return sg * am, e, 1
end

local function normalizeTuple(m: number, e: number, l: number): (number, number, number)
	if m ~= m or e ~= e or l ~= l then
		return 0, 1, -1
	end
	if l < 0 then
		return 0, 1, -1
	elseif l == huge then
		if m == 0 then return 0, 1, -1 end
		return (if m > 0 then 1 else -1), 1, huge
	elseif m == 0 then
		return 0, 0, 0
	end

	l = floor(l)
	if l == 0 then
		return fromNumberTuple(m)
	elseif l == 1 then
		return normalizeLayer1(m, e)
	end

	local sg = if m > 0 then 1 else -1
	if e == huge then
		return sg, 1, huge
	elseif e == -huge then
		if l == 2 then
			return 0, 0, 0
		end
		return sg, 1, l - 2
	end

	while l >= 2 and abs(e) <= INF_LIMIT do
		if l == 2 then
			local v = 10 ^ abs(e)
			return normalizeLayer1(sg, (if e < 0 then -v else v))
		else
			local es = if e > 0 then 1 else (if e < 0 then -1 else 0)
			e = es * (10 ^ abs(e))
			l -= 1
		end
	end

	return sg, e, l
end

-- Converts GammaNum 17-byte tuple (s, l17, e17) -> LightNum v1.20 (m, e, l)
local function fromLegacyTuple(s: number, l17: number, e17: number): (number, number, number)
	if s ~= s or l17 ~= l17 or e17 ~= e17 or l17 < 0 then
		return 0, 1, -1
	end
	if s == 0 then
		return 0, 0, 0
	end
	local sg = if s > 0 then 1 else -1
	if l17 == huge or e17 == huge then
		return sg, 1, huge
	end
	if l17 == 0 then
		return fromNumberTuple(sg * e17)
	elseif l17 == 1 then
		return normalizeLayer1(sg, e17)
	else
		return normalizeTuple(sg, e17, floor(l17))
	end
end

-- Converts LightNum v1.20 (m, e, l) -> GammaNum 17-byte tuple (s17, l17, e17)
local function toLegacyTuple(m: number, e: number, l: number): (number, number, number)
	if l < 0 or l ~= l or m ~= m or e ~= e then
		return 1, -1, 1
	elseif l == huge then
		return (if m >= 0 then 1 else -1), huge, 1
	elseif m == 0 then
		return 0, 0, 0
	end
	local sg = if m > 0 then 1 else -1
	if l == 0 then
		local am = if m > 0 then m else -m
		if am >= 1e10 or am <= 1e-10 then
			return sg, 1, log10(am)
		end
		return sg, 0, am
	elseif l == 1 then
		local totalE = if abs(e) < EXACT_INT_LIMIT then e + log10(abs(m)) else e
		if abs(totalE) >= 1e10 then
			local es = if totalE > 0 then 1 else -1
			local l17 = 2
			local e17 = es * log10(abs(totalE))
			while abs(e17) >= 1e10 do
				l17 += 1
				e17 = es * log10(abs(e17))
			end
			return sg, l17, e17
		end
		return sg, 1, totalE
	else
		local l17 = l
		local e17 = e
		while abs(e17) >= 1e10 do
			local es = if e17 > 0 then 1 else -1
			l17 += 1
			e17 = es * log10(abs(e17))
		end
		return sg, l17, e17
	end
end

local function pack(m: number, e: number, l: number): buffer
	local buf = bufCreate(25)
	local s = if l < 0 or m ~= m then 1 elseif m > 0 then 1 elseif m < 0 then -1 else 0
	bufWriteI8(buf, 0, s)
	bufWriteF64(buf, 1, l)
	bufWriteF64(buf, 9, e)
	bufWriteF64(buf, 17, m)
	return buf
end

local function pack17(m: number, e: number, l: number): buffer
	local s17, l17, e17 = toLegacyTuple(m, e, l)
	local buf = bufCreate(17)
	bufWriteI8(buf, 0, s17)
	bufWriteF64(buf, 1, l17)
	bufWriteF64(buf, 9, e17)
	return buf
end

local function writeBuf(buf: buffer, m: number, e: number, l: number): buffer
	if bufLen(buf) == 25 then
		local s = if l < 0 or m ~= m then 1 elseif m > 0 then 1 elseif m < 0 then -1 else 0
		bufWriteI8(buf, 0, s)
		bufWriteF64(buf, 1, l)
		bufWriteF64(buf, 9, e)
		bufWriteF64(buf, 17, m)
	else
		local s17, l17, e17 = toLegacyTuple(m, e, l)
		bufWriteI8(buf, 0, s17)
		bufWriteF64(buf, 1, l17)
		bufWriteF64(buf, 9, e17)
	end
	return buf
end

local function readAnyBuffer(buf: buffer): (number, number, number)
	local len = bufLen(buf)
	if len == 25 then
		return bufReadF64(buf, 17), bufReadF64(buf, 9), bufReadF64(buf, 1)
	elseif len == 17 then
		return fromLegacyTuple(bufReadI8(buf, 0), bufReadF64(buf, 1), bufReadF64(buf, 9))
	end
	error("Wrong Type: expected 25-byte or 17-byte LightNum buffer")
end

local function fastUnwrapAny(input: any): buffer
	local t = type(input)
	if t == "buffer" then
		local len = bufLen(input)
		if len == 25 or len == 17 then
			return input
		end
	elseif t == "table" then
		local b = input._buf
		if type(b) == "buffer" then
			local len = bufLen(b)
			if len == 25 or len == 17 then
				return b
			end
		end
	end
	error("Wrong Type: expected 17-byte or 25-byte LightNum buffer")
end

local parseStringTupleForward: (string) -> (number, number, number)

local function fastUnpack(input: any): (number, number, number)
	local t = type(input)
	if t == "table" then
		local m = input[1]
		local e = input[2]
		local l = input[3]
		if m ~= nil and e ~= nil and l ~= nil then
			return m, e, l
		end
		if input.m ~= nil and input.e ~= nil and input.l ~= nil then
			return normalizeTuple(input.m, input.e, input.l)
		end
		local b = input._buf
		if type(b) == "buffer" then
			return readAnyBuffer(b)
		end
		error("Wrong Type: invalid LightNum table or wrapper")
	elseif t == "number" then
		return fromNumberTuple(input)
	elseif t == "buffer" then
		return readAnyBuffer(input)
	elseif t == "string" then
		return parseStringTupleForward(input)
	else
		error("Wrong Type: expected LightNum table, buffer, number, string, or wrapper")
	end
end

local function compareMagnitude(m1: number, e1: number, l1: number, m2: number, e2: number, l2: number): number
	local eff1: number
	if l1 == 0 then
		eff1 = if abs(m1) >= 1 then 0 else -1
	elseif l1 == 1 then
		eff1 = if e1 >= 0 then 1 else -1
	else
		eff1 = if e1 > 0 then l1 else -l1
	end
	local eff2: number
	if l2 == 0 then
		eff2 = if abs(m2) >= 1 then 0 else -1
	elseif l2 == 1 then
		eff2 = if e2 >= 0 then 1 else -1
	else
		eff2 = if e2 > 0 then l2 else -l2
	end
	if eff1 < eff2 then return -1 end
	if eff1 > eff2 then return 1 end

	if l1 == 0 then
		if l2 == 1 then
			return if e2 >= 10 then -1 else 1
		end
		local a1, a2 = abs(m1), abs(m2)
		return if a1 < a2 then -1 elseif a1 > a2 then 1 else 0
	elseif l1 == 1 then
		if l2 == 0 then
			return if e1 >= 10 then 1 else -1
		end
		if e1 < e2 then return -1 end
		if e1 > e2 then return 1 end
		local a1, a2 = abs(m1), abs(m2)
		return if a1 < a2 then -1 elseif a1 > a2 then 1 else 0
	else
		if e1 < e2 then return -1 end
		if e1 > e2 then return 1 end
		return 0
	end
end

local function compareTuples(m1: number, e1: number, l1: number, m2: number, e2: number, l2: number): number?
	if l1 == 0 and l2 == 0 then
		return if m1 < m2 then -1 elseif m1 > m2 then 1 else 0
	elseif l1 == 1 and l2 == 1 then
		if m1 > 0 and m2 <= 0 then return 1 end
		if m1 < 0 and m2 >= 0 then return -1 end
		if m1 > 0 then
			if e1 < e2 then return -1 end
			if e1 > e2 then return 1 end
			return if m1 < m2 then -1 elseif m1 > m2 then 1 else 0
		else
			if e1 < e2 then return 1 end
			if e1 > e2 then return -1 end
			return if m1 < m2 then -1 elseif m1 > m2 then 1 else 0
		end
	end
	if l1 < 0 or l2 < 0 then
		return nil
	end
	if m1 == 0 then
		return if m2 > 0 then -1 elseif m2 < 0 then 1 else 0
	elseif m2 == 0 then
		return if m1 > 0 then 1 else -1
	end
	local s1 = if m1 > 0 then 1 else -1
	local s2 = if m2 > 0 then 1 else -1
	if s1 ~= s2 then
		return if s1 < s2 then -1 else 1
	end
	if l1 == huge and l2 == huge then return 0 end
	if l1 == huge then return s1 end
	if l2 == huge then return -s1 end
	local mag = compareMagnitude(m1, e1, l1, m2, e2, l2)
	return if s1 > 0 then mag else -mag
end

-- ============================================================================
-- CORE ARITHMETIC REGISTER-TUPLE KERNELS (FULL 53-BIT PRECISION)
-- ============================================================================

local function addTuple(m1: number, e1: number, l1: number, m2: number, e2: number, l2: number): (number, number, number)
	if l1 == 0 and l2 == 0 then
		local n = m1 + m2
		if n == 0 then
			return 0, 0, 0
		end
		local a = if n > 0 then n else -n
		if a < 1e10 and a > 1e-10 then
			return n, 0, 0
		end
		return fromNumberTuple(n)
	elseif l1 == 1 and l2 == 1 then
		if e1 < e2 or (e1 == e2 and abs(m1) < abs(m2)) then
			m1, m2 = m2, m1
			e1, e2 = e2, e1
		end
		local diff = e1 - e2
		if diff >= 17 then
			return m1, e1, 1
		end
		if diff == 0 then
			local m = m1 + m2
			if m == 0 then
				return 0, 0, 0
			end
			local am = if m > 0 then m else -m
			if am >= 1 and am < 10 then
				return m, e1, 1
			elseif am >= 10 then
				if e1 == 9 then
					return m * 1e9, 0, 0
				end
				return m * 0.1, e1 + 1, 1
			end
			return normalizeLayer1(m, e1)
		end
		local scale = negPow10[diff] or (10 ^ (-diff))
		local m = m1 + m2 * scale
		if m == 0 then
			return 0, 0, 0
		end
		local am = if m > 0 then m else -m
		if am >= 1 and am < 10 then
			if e1 == -10 and am > 1 then
				return m * 1e-10, 0, 0
			end
			return m, e1, 1
		end
		return normalizeLayer1(m, e1)
	end

	if l1 < 0 or l2 < 0 then
		return 0, 1, -1
	elseif l1 == huge or l2 == huge then
		if l1 == huge and l2 == huge and ((m1 > 0 and m2 < 0) or (m1 < 0 and m2 > 0)) then
			return 0, 1, -1
		end
		return (if l1 == huge then (if m1 > 0 then 1 else -1) else (if m2 > 0 then 1 else -1)), 1, huge
	elseif m1 == 0 then
		return m2, e2, l2
	elseif m2 == 0 then
		return m1, e1, l1
	elseif l1 == l2 and e1 == e2 and m1 == -m2 then
		return 0, 0, 0
	end

	if l1 >= 2 or l2 >= 2 then
		if compareMagnitude(m1, e1, l1, m2, e2, l2) >= 0 then
			return m1, e1, l1
		else
			return m2, e2, l2
		end
	end

	-- One is Layer 1, one is Layer 0
	if l1 == 0 then
		m1, e1, l1, m2, e2, l2 = m2, e2, l2, m1, e1, l1
	end
	if e1 >= 27 or (e1 <= -27 and abs(m2) >= 1e-10) then
		return (if e1 >= 27 then m1 else m2), (if e1 >= 27 then e1 else 0), (if e1 >= 27 then 1 else 0)
	end
	local m2n, e2n = fromNumberTuple(m2)
	return addTuple(m1, e1, 1, m2n, e2n, 1)
end

local function subTuple(m1: number, e1: number, l1: number, m2: number, e2: number, l2: number): (number, number, number)
	if l2 < 0 then return 0, 1, -1 end
	return addTuple(m1, e1, l1, -m2, e2, l2)
end

local function mulTuple(m1: number, e1: number, l1: number, m2: number, e2: number, l2: number): (number, number, number)
	if l1 == 0 and l2 == 0 then
		local n = m1 * m2
		if n == 0 then
			return 0, 0, 0
		end
		local a = if n > 0 then n else -n
		if a < 1e10 and a > 1e-10 then
			return n, 0, 0
		end
		return fromNumberTuple(n)
	elseif l1 == 1 and l2 == 1 then
		local m = m1 * m2
		local am = if m > 0 then m else -m
		local e = e1 + e2
		if am >= 10 then
			m *= 0.1
			e += 1
		end
		if abs(e) == huge then
			local scaled = e1 * 1e-300 + e2 * 1e-300
			if scaled == 0 then
				return m, 0, 0
			end
			local sg = if m > 0 then 1 else -1
			local es = if scaled > 0 then 1 else -1
			return sg, es * (300 + log10(abs(scaled))), 2
		end
		if e >= -10 and e < 10 then
			if e == -10 and am == 1 then
				return (if m > 0 then 1 else -1), -10, 1
			end
			local v = m * (pow10Arr[e + 11] or (10 ^ e))
			if v == 0 then return 0, 0, 0 end
			if abs(v) >= 1e10 then return (if v > 0 then 1 else -1), 10, 1 end
			if abs(v) <= 1e-10 then return (if v > 0 then 1 else -1), -10, 1 end
			return v, 0, 0
		end
		return m, e, 1
	end

	if l1 < 0 or l2 < 0 then
		return 0, 1, -1
	elseif (m1 == 0 and l2 == huge) or (m2 == 0 and l1 == huge) then
		return 0, 1, -1
	elseif l1 == huge or l2 == huge then
		return (if (m1 > 0) == (m2 > 0) then 1 else -1), 1, huge
	elseif m1 == 0 or m2 == 0 then
		return 0, 0, 0
	end

	-- Fast path: Layer 1 * Layer 0 (common multiplier loop: x * 1.0001)
	if l1 == 1 and l2 == 0 then
		local m = m1 * m2
		local am = if m > 0 then m else -m
		if am >= 1 and am < 10 then
			if e1 == -10 and am > 1 then return m * 1e-10, 0, 0 end
			return m, e1, 1
		elseif am >= 10 and am < 100 then
			m *= 0.1
			local e = e1 + 1
			if e >= -10 and e < 10 then
				local v = m * (pow10Arr[e + 11] or (10 ^ e))
				if abs(v) >= 1e10 then return (if v > 0 then 1 else -1), 10, 1 end
				if abs(v) <= 1e-10 then return (if v > 0 then 1 else -1), -10, 1 end
				return v, 0, 0
			end
			return m, e, 1
		end
		return normalizeLayer1(m, e1)
	elseif l1 == 0 and l2 == 1 then
		local m = m1 * m2
		local am = if m > 0 then m else -m
		if am >= 1 and am < 10 then
			if e2 == -10 and am > 1 then return m * 1e-10, 0, 0 end
			return m, e2, 1
		elseif am >= 10 and am < 100 then
			m *= 0.1
			local e = e2 + 1
			if e >= -10 and e < 10 then
				local v = m * (pow10Arr[e + 11] or (10 ^ e))
				if abs(v) >= 1e10 then return (if v > 0 then 1 else -1), 10, 1 end
				if abs(v) <= 1e-10 then return (if v > 0 then 1 else -1), -10, 1 end
				return v, 0, 0
			end
			return m, e, 1
		end
		return normalizeLayer1(m, e2)
	end

	local rs = if (m1 > 0) == (m2 > 0) then 1 else -1
	if l1 >= 3 or l2 >= 3 then
		if l1 > l2 then
			return rs, e1, l1
		elseif l1 < l2 then
			return rs, e2, l2
		elseif abs(e1) >= abs(e2) then
			return rs, e1, l1
		else
			return rs, e2, l2
		end
	end

	-- l1 == 2 or l2 == 2 (value = 10^(±10^|e|))
	if l1 == 2 and l2 == 2 and e1 == -e2 then
		return rs, 0, 0
	end
	if l1 == 2 and l2 < 2 then
		return rs, e1, 2
	elseif l2 == 2 and l1 < 2 then
		return rs, e2, 2
	else
		local s1, s2 = (if e1 >= 0 then 1 else -1), (if e2 >= 0 then 1 else -1)
		local ls1, le1, ll1 = normalizeLayer1(s1, abs(e1))
		local ls2, le2, ll2 = normalizeLayer1(s2, abs(e2))
		local sm, se, sl = addTuple(ls1, le1, ll1, ls2, le2, ll2)
		if sm == 0 then return rs, 0, 0 end
		local sgE = if sm > 0 then 1 else -1
		local expLog = if sl == 0 then log10(abs(sm)) else se + log10(abs(sm))
		return normalizeTuple(rs, sgE * expLog, sl + 1)
	end
end

local function divTuple(m1: number, e1: number, l1: number, m2: number, e2: number, l2: number): (number, number, number)
	if l1 == 0 and l2 == 0 then
		if m2 == 0 then
			if m1 == 0 then return 0, 1, -1 end
			return (if m1 > 0 then 1 else -1), 1, huge
		elseif m1 == 0 then
			return 0, 0, 0
		end
		local n = m1 / m2
		if n == 0 then return 0, 0, 0 end
		local a = if n > 0 then n else -n
		if a < 1e10 and a > 1e-10 then
			return n, 0, 0
		end
		return fromNumberTuple(n)
	elseif l1 == 1 and l2 == 1 then
		local m = m1 / m2
		local am = if m > 0 then m else -m
		local e = e1 - e2
		if am < 1 then
			m *= 10
			am *= 10
			e -= 1
		end
		if abs(e) == huge then
			local scaled = e1 * 1e-300 - e2 * 1e-300
			if scaled == 0 then return m, 0, 0 end
			local sg = if m > 0 then 1 else -1
			local es = if scaled > 0 then 1 else -1
			return sg, es * (300 + log10(abs(scaled))), 2
		end
		if e >= -10 and e < 10 then
			if e == -10 and am == 1 then
				return (if m > 0 then 1 else -1), -10, 1
			end
			local v = m * (pow10Arr[e + 11] or (10 ^ e))
			if v == 0 then return 0, 0, 0 end
			if abs(v) >= 1e10 then return (if v > 0 then 1 else -1), 10, 1 end
			if abs(v) <= 1e-10 then return (if v > 0 then 1 else -1), -10, 1 end
			return v, 0, 0
		end
		return m, e, 1
	end

	if l1 < 0 or l2 < 0 then
		return 0, 1, -1
	elseif m2 == 0 then
		if m1 == 0 then return 0, 1, -1 end
		return (if m1 > 0 then 1 else -1), 1, huge
	elseif l1 == huge and l2 == huge then
		return 0, 1, -1
	elseif l1 == huge then
		return (if (m1 > 0) == (m2 > 0) then 1 else -1), 1, huge
	elseif l2 == huge or m1 == 0 then
		return 0, 0, 0
	end

	if l1 == 1 and l2 == 0 then
		return normalizeLayer1(m1 / m2, e1)
	elseif l1 == 0 and l2 == 1 then
		return normalizeLayer1(m1 / m2, -e2)
	end

	local rs = if (m1 > 0) == (m2 > 0) then 1 else -1
	if l1 >= 3 or l2 >= 3 then
		if l1 > l2 then
			return rs, e1, l1
		elseif l1 < l2 then
			return rs, -e2, l2
		elseif abs(e1) >= abs(e2) then
			return rs, e1, l1
		else
			return rs, -e2, l2
		end
	end

	if l1 == 2 and l2 == 2 and e1 == e2 then
		return rs, 0, 0
	end
	if l1 == 2 and l2 < 2 then
		return rs, e1, 2
	elseif l2 == 2 and l1 < 2 then
		return rs, -e2, 2
	else
		local s1, s2 = (if e1 >= 0 then 1 else -1), (if e2 >= 0 then 1 else -1)
		local ls1, le1, ll1 = normalizeLayer1(s1, abs(e1))
		local ls2, le2, ll2 = normalizeLayer1(s2, abs(e2))
		local sm, se, sl = subTuple(ls1, le1, ll1, ls2, le2, ll2)
		if sm == 0 then return rs, 0, 0 end
		local sgE = if sm > 0 then 1 else -1
		local expLog = if sl == 0 then log10(abs(sm)) else se + log10(abs(sm))
		return normalizeTuple(rs, sgE * expLog, sl + 1)
	end
end

local function recipTuple(m: number, e: number, l: number): (number, number, number)
	if l < 0 then return 0, 1, -1 end
	if m == 0 then return 1, 1, huge end
	if l == huge then return 0, 0, 0 end
	if l == 0 then return fromNumberTuple(1 / m) end
	if l == 1 then return normalizeLayer1(1 / m, -e) end
	return (if m > 0 then 1 else -1), -e, l
end

local function floorTuple(m: number, e: number, l: number): (number, number, number)
	if l < 0 or l == huge or m == 0 then return m, e, l end
	if l == 0 then
		return fromNumberTuple(floor(m))
	elseif l == 1 then
		if e < 0 then
			return (if m > 0 then 0 else -1), 0, 0
		elseif e <= 15 then
			return fromNumberTuple(floor(m * (posPow10[e] or (10 ^ e))))
		end
		return m, e, 1
	else
		if e < 0 then
			return (if m > 0 then 0 else -1), 0, 0
		end
		return m, e, l
	end
end

local function ceilTuple(m: number, e: number, l: number): (number, number, number)
	if l < 0 or l == huge or m == 0 then return m, e, l end
	if l == 0 then
		return fromNumberTuple(ceil(m))
	elseif l == 1 then
		if e < 0 then
			return (if m > 0 then 1 else 0), 0, 0
		elseif e <= 15 then
			return fromNumberTuple(ceil(m * (posPow10[e] or (10 ^ e))))
		end
		return m, e, 1
	else
		if e < 0 then
			return (if m > 0 then 1 else 0), 0, 0
		end
		return m, e, l
	end
end

local function roundTuple(m: number, e: number, l: number): (number, number, number)
	if l < 0 or l == huge or m == 0 then return m, e, l end
	if l == 0 then
		return fromNumberTuple(round(m))
	elseif l == 1 then
		if e < 0 then
			return 0, 0, 0
		elseif e <= 15 then
			return fromNumberTuple(round(m * (posPow10[e] or (10 ^ e))))
		end
		return m, e, 1
	else
		if e < 0 then return 0, 0, 0 end
		return m, e, l
	end
end

local function intdivTuple(m1: number, e1: number, l1: number, m2: number, e2: number, l2: number): (number, number, number)
	local qm, qe, ql = divTuple(m1, e1, l1, m2, e2, l2)
	return floorTuple(qm, qe, ql)
end

local function modTuple(m1: number, e1: number, l1: number, m2: number, e2: number, l2: number): (number, number, number)
	if l1 < 0 or l2 < 0 or m2 == 0 or l1 == huge then return 0, 1, -1 end
	if m1 == 0 then return 0, 0, 0 end
	if l2 == huge then return m1, e1, l1 end
	if l1 == 0 and l2 == 0 then
		return fromNumberTuple(m1 % m2)
	end
	local qm, qe, ql = divTuple(m1, e1, l1, m2, e2, l2)
	qm, qe, ql = floorTuple(qm, qe, ql)
	local pm, pe, pl = mulTuple(qm, qe, ql, m2, e2, l2)
	return subTuple(m1, e1, l1, pm, pe, pl)
end

local function fmaTuple(m1: number, e1: number, l1: number, m2: number, e2: number, l2: number, m3: number, e3: number, l3: number): (number, number, number)
	if l1 == 0 and l2 == 0 and l3 == 0 then
		local p = m1 * m2
		local ap = if p > 0 then p else -p
		if ap < 1e10 and (ap > 1e-10 or p == 0) then
			local n = p + m3
			if n == 0 then return 0, 0, 0 end
			local an = if n > 0 then n else -n
			if an < 1e10 and an > 1e-10 then
				return n, 0, 0
			end
		end
	elseif l1 == 1 and l2 == 1 and l3 == 1 then
		local pm = m1 * m2
		local pam = if pm > 0 then pm else -pm
		local pe = e1 + e2
		if pam >= 10 then
			pm *= 0.1
			pe += 1
		end
		if (pe >= 10 or pe < -10) and abs(pe) < huge then
			if pe < e3 or (pe == e3 and abs(pm) < abs(m3)) then
				pm, m3 = m3, pm
				pe, e3 = e3, pe
			end
			local diff = pe - e3
			if diff >= 17 then
				return pm, pe, 1
			end
			if diff == 0 then
				local m = pm + m3
				if m == 0 then return 0, 0, 0 end
				local am = if m > 0 then m else -m
				if am >= 1 and am < 10 then
					return m, pe, 1
				elseif am >= 10 then
					if pe == 9 then return m * 1e9, 0, 0 end
					return m * 0.1, pe + 1, 1
				end
				return normalizeLayer1(m, pe)
			end
			local scale = negPow10[diff] or (10 ^ (-diff))
			local m = pm + m3 * scale
			if m == 0 then return 0, 0, 0 end
			local am = if m > 0 then m else -m
			if am >= 1 and am < 10 then
				if pe == -10 and am > 1 then
					return m * 1e-10, 0, 0
				end
				return m, pe, 1
			end
			return normalizeLayer1(m, pe)
		end
	end
	local pm, pe, pl = mulTuple(m1, e1, l1, m2, e2, l2)
	return addTuple(pm, pe, pl, m3, e3, l3)
end

local function abslog10Tuple(m: number, e: number, l: number): (number, number, number)
	if l < 0 or m == 0 then
		return 0, 1, -1
	elseif l == huge then
		return 1, 1, huge
	elseif l == 0 then
		local am = abs(m)
		if am == 1 then return 0, 0, 0 end
		return fromNumberTuple(log10(am))
	elseif l == 1 then
		local v = e + log10(abs(m))
		return fromNumberTuple(v)
	elseif l == 2 then
		return normalizeLayer1((if e > 0 then 1 else -1), abs(e))
	else
		return (if e > 0 then 1 else -1), abs(e), l - 1
	end
end

local function log10Tuple(m: number, e: number, l: number): (number, number, number)
	if l < 0 or m <= 0 then return 0, 1, -1 end
	return abslog10Tuple(m, e, l)
end

local function pow10Tuple(m: number, e: number, l: number): (number, number, number)
	if l < 0 then
		return 0, 1, -1
	elseif l == huge then
		if m < 0 then return 0, 0, 0 end
		return 1, 1, huge
	elseif m == 0 then
		return 1, 0, 0
	elseif l == 0 then
		if abs(m) <= 307 then
			local v = 10 ^ m
			if v == 0 then return 0, 0, 0 end
			return fromNumberTuple(v)
		end
		return normalizeLayer1(1, m)
	elseif l == 1 then
		if e < 0 then
			if e >= -16 then
				return fromNumberTuple(1 + expm1((m * (10 ^ e)) * LN_10))
			end
			return 1, 0, 0
		end
		local magLog = e + log10(abs(m))
		if magLog <= INF_LIMIT then
			local newE = m * (10 ^ e)
			if abs(newE) < huge then
				return normalizeLayer1(1, newE)
			end
		end
		return normalizeTuple(1, (if m > 0 then magLog else -magLog), 2)
	else
		if e < 0 then
			return 1, 0, 0
		elseif m < 0 then
			return 0, 0, 0
		end
		return 1, e, l + 1
	end
end

local function powTuple(m1: number, e1: number, l1: number, m2: number, e2: number, l2: number): (number, number, number)
	if l1 < 0 or l2 < 0 then return 0, 1, -1 end
	if m2 == 0 then return 1, 0, 0 end
	if m1 == 0 then
		if m2 < 0 then return 1, 1, huge end
		return 0, 0, 0
	end

	local resultSign = 1
	if m1 < 0 then
		if l2 >= 2 or (l2 == 0 and m2 ~= floor(m2)) then
			return 0, 1, -1
		end
		if l2 == 0 then
			if abs(m2) % 2 == 1 then resultSign = -1 end
		elseif e2 < 0 then
			return 0, 1, -1
		elseif e2 <= 15 then
			local intVal = abs(m2) * (posPow10[e2] or (10 ^ e2))
			if intVal % 2 == 1 then resultSign = -1 end
		end
	end

	if l1 == 0 and l2 == 0 then
		local nativeRes = m1 ^ m2
		if nativeRes == nativeRes and abs(nativeRes) < huge and nativeRes ~= 0 then
			return fromNumberTuple(nativeRes)
		end
	end

	if l1 == huge then
		if m2 < 0 then return 0, 0, 0 end
		return resultSign, 1, huge
	elseif l2 == huge then
		local cmp = compareMagnitude(m1, e1, l1, 1, 0, 0)
		if cmp == 0 then
			return 1, 0, 0
		elseif (cmp > 0 and m2 > 0) or (cmp < 0 and m2 < 0) then
			return 1, 1, huge
		else
			return 0, 0, 0
		end
	end

	if l1 == 0 and m1 == 1 then return 1, 0, 0 end
	if l2 == 0 and m2 == 1 then return m1, e1, l1 end

	-- Full 53-bit precision fast path for Layer 1 ^ Layer 0
	if l1 == 1 and l2 == 0 then
		local am1 = abs(m1)
		local p = m2
		local ep = e1 * p
		if abs(ep) < huge then
			if p == floor(p) and abs(p) <= 20 and (abs(e1) <= 1e14 or abs(e1) >= EXACT_INT_LIMIT) then
				return normalizeLayer1(resultSign * (am1 ^ p), ep)
			end
			if abs(ep) <= 1e13 and abs(p) <= 1e6 then
				local epInt = floor(ep)
				local epFrac = ep - epInt
				local lp = log10(am1) * p + epFrac
				local k = floor(lp)
				local rem = lp - k
				return normalizeLayer1(resultSign * (10 ^ rem), epInt + k)
			elseif abs(ep) >= EXACT_INT_LIMIT and abs(p) < 300 then
				return normalizeLayer1(resultSign * (am1 ^ p), ep)
			end
		end
	end

	local lm, le, ll = abslog10Tuple(m1, e1, l1)
	local mm, me, ml = mulTuple(lm, le, ll, m2, e2, l2)
	local pm, pe, pl = pow10Tuple(mm, me, ml)
	if pl < 0 then return 0, 1, -1 end
	if pm == 0 then return 0, 0, 0 end
	return resultSign * abs(pm), pe, pl
end

local function rootTuple(m1: number, e1: number, l1: number, m2: number, e2: number, l2: number): (number, number, number)
	if l1 < 0 or l2 < 0 or m2 == 0 then return 0, 1, -1 end
	if m1 == 0 then
		if m2 < 0 then return 1, 1, huge end
		return 0, 0, 0
	end
	local resultSign = 1
	if m1 < 0 then
		if l2 ~= 0 or m2 ~= floor(m2) or abs(m2) % 2 ~= 1 then
			return 0, 1, -1
		end
		resultSign = -1
	end
	if l1 == huge then
		if m2 < 0 then return 0, 0, 0 end
		return resultSign, 1, huge
	elseif l2 == huge then
		return 1, 0, 0
	end
	if l1 == 0 and abs(m1) == 1 then return resultSign, 0, 0 end
	if l2 == 0 and m2 == 1 then return m1, e1, l1 end

	local lm, le, ll = abslog10Tuple(m1, e1, l1)
	local dm, de, dl = divTuple(lm, le, ll, m2, e2, l2)
	local pm, pe, pl = pow10Tuple(dm, de, dl)
	if pl < 0 then return 0, 1, -1 end
	if pm == 0 then return 0, 0, 0 end
	return resultSign * abs(pm), pe, pl
end

local function sqrtTuple(m: number, e: number, l: number): (number, number, number)
	if l < 0 or m < 0 then return 0, 1, -1 end
	if l == huge then return 1, 1, huge end
	if m == 0 then return 0, 0, 0 end
	if l == 0 then return fromNumberTuple(math.sqrt(m)) end
	if l == 1 then
		local q = e * 0.5
		local iq = floor(q)
		local fq = q - iq
		local sm = math.sqrt(m) * (if fq ~= 0 then 10 ^ fq else 1)
		return normalizeLayer1(sm, iq)
	end
	local lm, le, ll = abslog10Tuple(m, e, l)
	local dm, de, dl = divTuple(lm, le, ll, 2, 0, 0)
	return pow10Tuple(dm, de, dl)
end

local function cbrtTuple(m: number, e: number, l: number): (number, number, number)
	if l < 0 then return 0, 1, -1 end
	if l == huge then return (if m > 0 then 1 else -1), 1, huge end
	if m == 0 then return 0, 0, 0 end
	if l == 0 then
		local s = if m > 0 then 1 else -1
		return fromNumberTuple(s * (abs(m) ^ (1 / 3)))
	end
	local rs = if m > 0 then 1 else -1
	local lm, le, ll = abslog10Tuple(m, e, l)
	local dm, de, dl = divTuple(lm, le, ll, 3, 0, 0)
	local pm, pe, pl = pow10Tuple(dm, de, dl)
	if pm == 0 then return 0, 0, 0 end
	return rs * abs(pm), pe, pl
end

local function pow2Tuple(m: number, e: number, l: number): (number, number, number)
	if l < 0 then return 0, 1, -1 end
	if l == huge then return (if m < 0 then 0 else 1), (if m < 0 then 0 else 1), (if m < 0 then 0 else huge) end
	if m == 0 then return 1, 0, 0 end
	if l == 0 and abs(m) < 1000 then
		local v = 2 ^ m
		if v == 0 then return 0, 0, 0 end
		if abs(v) < huge then return fromNumberTuple(v) end
	end
	local mm, me, ml = mulTuple(m, e, l, LOG10_2, 0, 0)
	return pow10Tuple(mm, me, ml)
end

local function expTuple(m: number, e: number, l: number): (number, number, number)
	if l < 0 then return 0, 1, -1 end
	if l == huge then return (if m < 0 then 0 else 1), (if m < 0 then 0 else 1), (if m < 0 then 0 else huge) end
	if m == 0 then return 1, 0, 0 end
	if l == 0 and m == 1 then return EULER, 0, 0 end
	if l == 0 and abs(m) < 700 then
		local v = mexp(m)
		if v == 0 then return 0, 0, 0 end
		if abs(v) < huge then return fromNumberTuple(v) end
	end
	local mm, me, ml = mulTuple(m, e, l, INV_LN_10, 0, 0)
	return pow10Tuple(mm, me, ml)
end

local function logTuple(m1: number, e1: number, l1: number, m2: number, e2: number, l2: number): (number, number, number)
	if l1 < 0 or l2 < 0 or m1 <= 0 or m2 <= 0 or (l2 == 0 and m2 == 1) then
		return 0, 1, -1
	elseif l2 == huge then
		if l1 == huge then return 0, 1, -1 end
		return 0, 0, 0
	elseif l1 == huge then
		local baseVsOne = compareMagnitude(m2, e2, l2, 1, 0, 0)
		return (if baseVsOne > 0 then 1 else -1), 1, huge
	elseif l1 == 0 and m1 == 1 then
		return 0, 0, 0
	elseif l1 == 0 and l2 == 0 then
		return fromNumberTuple(log10(m1) / log10(m2))
	end
	local a1m, a1e, a1l = log10Tuple(m1, e1, l1)
	local a2m, a2e, a2l = log10Tuple(m2, e2, l2)
	return divTuple(a1m, a1e, a1l, a2m, a2e, a2l)
end

local function log2Tuple(m: number, e: number, l: number): (number, number, number)
	if l < 0 or m <= 0 then return 0, 1, -1 end
	if l == huge then return 1, 1, huge end
	if l == 0 then
		if m == 1 then return 0, 0, 0 end
		return fromNumberTuple(log10(m) / LOG10_2)
	end
	local lm, le, ll = log10Tuple(m, e, l)
	return divTuple(lm, le, ll, LOG10_2, 0, 0)
end

local function lnTuple(m: number, e: number, l: number): (number, number, number)
	if l < 0 or m <= 0 then return 0, 1, -1 end
	if l == huge then return 1, 1, huge end
	if l == 0 then
		if m == 1 then return 0, 0, 0 end
		return fromNumberTuple(mlog(m))
	end
	local lm, le, ll = log10Tuple(m, e, l)
	return divTuple(lm, le, ll, INV_LN_10, 0, 0)
end

local function tetrTuple(m1: number, e1: number, l1: number, height: number): (number, number, number)
	if l1 < 0 or height ~= height or abs(height) == huge or height < 0 then
		return 0, 1, -1
	end
	if height == 0 then return 1, 0, 0 end
	if height == 1 then return m1, e1, l1 end

	local isInt = (height == floor(height))
	if not isInt and m1 <= 0 then return 0, 1, -1 end
	if l1 == huge then
		if m1 > 0 then return 1, 1, huge end
		return 0, 1, -1
	end
	if l1 == 0 and m1 == 1 then return 1, 0, 0 end

	if isInt and m1 <= 0 then
		if l1 == 0 and m1 == -1 then return -1, 0, 0 end
		local rm, re, rl = m1, e1, l1
		for _ = 2, height do
			rm, re, rl = powTuple(m1, e1, l1, rm, re, rl)
			if rl < 0 then return 0, 1, -1 end
			if rl == huge and m1 > 0 then return 1, 1, huge end
		end
		return rm, re, rl
	end

	local whole = floor(height)
	local frac = height - whole
	local rm, re, rl
	local startIdx: number
	if frac == 0 then
		rm, re, rl = m1, e1, l1
		startIdx = 2
	else
		rm, re, rl = powTuple(m1, e1, l1, frac, 0, 0)
		if rl < 0 or rl == huge then return rm, re, rl end
		startIdx = 1
	end

	local baseGt1 = compareTuples(m1, e1, l1, 1, 0, 0) == 1
	local logBaseM, logBaseE, logBaseL = 0, 0, 0
	if baseGt1 then
		logBaseM, logBaseE, logBaseL = abslog10Tuple(m1, e1, l1)
	end

	for i = startIdx, whole do
		local prevM, prevE, prevL = rm, re, rl
		rm, re, rl = powTuple(m1, e1, l1, rm, re, rl)
		if rl < 0 or rl == huge then return rm, re, rl end
		if rm == prevM and re == prevE and rl == prevL then return rm, re, rl end
		if baseGt1 and i < whole and rl >= 3 and re > 0 then
			if compareMagnitude(rm, re, rl, logBaseM, logBaseE, logBaseL) > 0 then
				return 1, re, rl + (whole - i)
			end
		end
	end
	return rm, re, rl
end

local function slogTuple(m1: number, e1: number, l1: number, bm: number, be: number, bl: number): number
	if l1 < 0 or bl < 0 or bm <= 0 or (bl == 0 and bm <= 1) then
		return 0 / 0
	end
	if bl == 0 and bm <= E_INV_E + 1e-15 then
		if l1 == huge or compareTuples(m1, e1, l1, EULER, 0, 0) == 1 then
			return 0 / 0
		end
	end
	if l1 == huge then return huge end
	if m1 < 0 then return 0 / 0 end
	if m1 == 0 then return -1 end

	local cmpInit = compareTuples(m1, e1, l1, 1, 0, 0)
	if cmpInit == 0 then
		return 0
	elseif cmpInit == -1 then
		local lm, le, ll = logTuple(m1, e1, l1, bm, be, bl)
		if ll == 0 then return lm end
		if ll == 1 and le < 0 and le >= -308 then return lm * (10 ^ le) end
		return -1
	end

	local steps = 0
	local cm, ce, cl = m1, e1, l1

	local lbm, lbe, lbl = log10Tuple(bm, be, bl)
	if cl >= 5 and ce > 0 and compareMagnitude(cm, ce, cl, lbm, lbe, lbl + 3) > 0 then
		local jump = cl - mmax(4, lbl + 3)
		cl -= jump
		steps += jump
	end

	local guard = 0
	while compareTuples(cm, ce, cl, 1, 0, 0) == 1 and guard < 10000 do
		local nm, ne, nl = logTuple(cm, ce, cl, bm, be, bl)
		if nl < 0 then return 0 / 0 end
		local cmpPrev = compareTuples(nm, ne, nl, cm, ce, cl)
		if cmpPrev == nil or cmpPrev >= 0 then
			return 0 / 0
		end
		cm, ce, cl = nm, ne, nl
		steps += 1
		guard += 1
	end

	if guard >= 10000 then return 0 / 0 end

	local val = 0
	if cm > 0 then
		if cl == 0 then
			val = cm
		elseif cl == 1 and ce < 0 and ce >= -308 then
			val = cm * (10 ^ ce)
		end
	end
	if steps == 0 and val == 1 then return 0 end
	return (steps - 1) + val
end

-- Lanczos Gamma & Stirling Series
local LANCZOS_C = {
	0.99999999999980993,
	676.5203681218851,
	-1259.1392167224028,
	771.32342877765313,
	-176.61502916214059,
	12.507343278686905,
	-0.13857109526572012,
	9.9843695780195716e-6,
	1.5056327351493116e-7,
}

local function lanczosGamma(n: number): number
	if n > 171.6236 then return 1.8e308 end
	if n >= 0.5 then
		n -= 1
		local x = LANCZOS_C[1]
		for i = 1, 8 do
			x += LANCZOS_C[i + 1] / (n + i)
		end
		local t = n + 7.5
		return x * (t ^ (n + 0.5 - 36)) * mexp(-t) * (t ^ 36) * SQRT_TWO_PI
	end
	return pi / (msin(pi * n) * lanczosGamma(1 - n))
end

local function gammaTuple(m: number, e: number, l: number): (number, number, number)
	if l < 0 or m == 0 then
		return 0, 1, -1
	elseif m < 0 then
		if l ~= 0 or m == floor(m) then
			return 0, 1, -1
		end
		local am = -m
		if am <= 24 then
			return fromNumberTuple(lanczosGamma(m))
		end
		local sinTerm = -msin(pi * (am % 2))
		if sinTerm == 0 then return 0, 1, -1 end
		local gm, ge, gl = gammaTuple(1 + am, 0, 0)
		local dm, de, dl = fromNumberTuple(sinTerm)
		local pm, pe, pl = mulTuple(dm, de, dl, gm, ge, gl)
		return divTuple(pi, 0, 0, pm, pe, pl)
	elseif l == huge then
		return 1, 1, huge
	elseif l == 0 then
		if m <= 24 then
			return fromNumberTuple(lanczosGamma(m))
		end
		local t = m - 1
		local l1 = (HALF_LN_TWO_PI + (t + 0.5) * mlog(t)) - t
		local n2 = t * t
		local np = t
		l1 += 1 / (12 * np)
		np *= n2
		l1 -= 1 / (360 * np)
		np *= n2
		l1 += 1 / (1260 * np)
		np *= n2
		l1 -= 1 / (1680 * np)
		return expTuple(l1, 0, 0)
	elseif l == 1 then
		if e < 0 then
			return normalizeLayer1(1 / m, -e)
		end
		local lnm, lne, lnl = lnTuple(m, e, 1)
		local nhm, nhe, nhl = subTuple(m, e, 1, 0.5, 0, 0)
		local p1m, p1e, p1l = mulTuple(nhm, nhe, nhl, lnm, lne, lnl)
		local p2m, p2e, p2l = subTuple(p1m, p1e, p1l, m, e, 1)
		local p3m, p3e, p3l = addTuple(p2m, p2e, p2l, HALF_LN_TWO_PI, 0, 0)
		return expTuple(p3m, p3e, p3l)
	else
		if e < 0 then return 1, -e, l end
		return expTuple(m, e, l)
	end
end

local function geosumTuple(bm: number, be: number, bl: number, mm: number, me: number, ml: number, sm: number, se: number, sl: number, lm: number, le: number, ll: number): (number, number, number)
	if compareTuples(lm, le, ll, sm, se, sl) == -1 then
		return 0, 0, 0
	end
	if ml == 0 and mm == 1 then
		local dm, de, dl = subTuple(lm, le, ll, sm, se, sl)
		local cm, ce, cl = addTuple(dm, de, dl, 1, 0, 0)
		return mulTuple(cm, ce, cl, bm, be, bl)
	end
	if bl == 0 and ml == 0 and sl == 0 and ll == 0 and bm >= 0 and mm > 0 and abs(mm - 1) >= 1e-6 then
		local p1 = mm ^ (lm + 1)
		local p2 = if sm == 0 then 1 else mm ^ sm
		if p1 < 1e290 and p2 < 1e290 and (p1 > 1e-290 or p1 == 0) and (p2 > 1e-290 or p2 == 0) then
			local res = bm * ((p1 - p2) / (mm - 1))
			if res == res and abs(res) < 1e300 and res ~= 0 then
				return fromNumberTuple(res)
			end
		end
	end
	local l1m, l1e, l1l = addTuple(lm, le, ll, 1, 0, 0)
	local p1m, p1e, p1l = powTuple(mm, me, ml, l1m, l1e, l1l)
	local p2m, p2e, p2l
	if sm == 0 then
		p2m, p2e, p2l = 1, 0, 0
	else
		p2m, p2e, p2l = powTuple(mm, me, ml, sm, se, sl)
	end
	local numM, numE, numL = subTuple(p1m, p1e, p1l, p2m, p2e, p2l)
	local prodM, prodE, prodL = mulTuple(numM, numE, numL, bm, be, bl)
	local denM, denE, denL = subTuple(mm, me, ml, 1, 0, 0)
	return divTuple(prodM, prodE, prodL, denM, denE, denL)
end

local function geosumRTuple(bm: number, be: number, bl: number, mm: number, me: number, ml: number, sm: number, se: number, sl: number, amm: number, ame: number, aml: number): (number, number, number)
	if compareTuples(mm, me, ml, 1, 0, 0) == 0 then
		local dm, de, dl = divTuple(amm, ame, aml, bm, be, bl)
		local am, ae, al = addTuple(dm, de, dl, sm, se, sl)
		return subTuple(am, ae, al, 1, 0, 0)
	end
	local dmM, dmE, dmL = subTuple(mm, me, ml, 1, 0, 0)
	local numM, numE, numL = mulTuple(dmM, dmE, dmL, amm, ame, aml)
	local pM, pE, pL = powTuple(mm, me, ml, sm, se, sl)
	local denM, denE, denL = mulTuple(bm, be, bl, pM, pE, pL)
	local scM, scE, scL = divTuple(numM, numE, numL, denM, denE, denL)
	local p1M, p1E, p1L = addTuple(scM, scE, scL, 1, 0, 0)
	local lgM, lgE, lgL = logTuple(p1M, p1E, p1L, mm, me, ml)
	local am, ae, al = addTuple(lgM, lgE, lgL, sm, se, sl)
	return subTuple(am, ae, al, 1, 0, 0)
end

-- ============================================================================
-- PUBLIC CONVERSION, TABLE & BUFFER API
-- ============================================================================

function ln.new(s: number?, l: number?, e: number?): buffer
	if s == nil then return bufCreate(25) end
	if type(s) ~= "number" then error("Wrong Type: new(), Input 1") end
	if s ~= s then return pack(0, 1, -1) end
	if s == 0 then return bufCreate(25) end
	local layer = if l == nil then 0 else l
	local expVal = if e == nil then 0 else e
	if type(layer) ~= "number" then error("Wrong Type: new(), Input 2") end
	if type(expVal) ~= "number" then error("Wrong Type: new(), Input 3") end
	return pack(fromLegacyTuple(s, layer, expVal))
end

function ln.createCheckless(s: number, l: number, e: number): buffer
	local buf = bufCreate(17)
	bufWriteI8(buf, 0, s)
	bufWriteF64(buf, 1, l)
	bufWriteF64(buf, 9, e)
	return buf
end

function ln.fromNumber(n: number): buffer
	if type(n) ~= "number" then return pack(0, 1, -1) end
	return pack(fromNumberTuple(n))
end

local function fromScientificTuple(m: number, e: number): (number, number, number)
	if m ~= m or e ~= e then return 0, 1, -1 end
	if m == 0 then return 0, 0, 0 end
	local sg = if m > 0 then 1 else -1
	if abs(m) == huge then
		if e == -huge then return 0, 1, -1 end
		return sg, 1, huge
	elseif e == huge then
		return sg, 1, huge
	elseif e == -huge then
		return 0, 0, 0
	end
	return normalizeLayer1(m, e)
end

function ln.fromScientific(m: number, e: number): buffer
	if type(m) ~= "number" then error("Wrong Type: fromScientific(), Input 1") end
	if type(e) ~= "number" then error("Wrong Type: fromScientific(), Input 2") end
	return pack(fromScientificTuple(m, e))
end

local function trimFast(s: string): string
	local len = #s
	if len == 0 then return s end
	if sbyte(s, 1) > 32 and sbyte(s, len) > 32 then return s end
	local i = 1
	while i <= len and sbyte(s, i) <= 32 do
		i += 1
	end
	local j = len
	while j >= i and sbyte(s, j) <= 32 do
		j -= 1
	end
	if i > j then return "" end
	return s:sub(i, j)
end

local function fromStringTuple(str: string): (number, number, number)
	if type(str) ~= "string" then
		return 0, 1, -1
	end
	str = trimFast(str)
	local len = #str
	if len == 0 then
		return 0, 0, 0
	end

	local b1 = sbyte(str, 1)
	if len <= 9 and (b1 == 105 or b1 == 73 or b1 == 110 or b1 == 78 or ((b1 == 43 or b1 == 45) and len >= 4)) then
		local lower = str:lower()
		if lower == "inf" or lower == "+inf" or lower == "infinity" or lower == "+infinity" then
			return 1, 1, huge
		elseif lower == "-inf" or lower == "-infinity" then
			return -1, 1, huge
		elseif lower == "nan" then
			return 0, 1, -1
		end
	end

	local semicolon = string.find(str, ";", 1, true)
	if semicolon then
		local left = trimFast(str:sub(1, semicolon - 1))
		local right = trimFast(str:sub(semicolon + 1))
		if #left == 0 or #right == 0 or string.find(right, ";", 1, true) then
			return 0, 1, -1
		end
		local layerRaw = tonumber(left)
		local exponent = tonumber(right)
		if layerRaw == nil or exponent == nil or layerRaw ~= layerRaw or exponent ~= exponent then
			return 0, 1, -1
		end
		local s = sign(layerRaw)
		if s == 0 then s = 1 end
		return fromLegacyTuple(s, abs(layerRaw), exponent)
	end

	local caretPos = string.find(str, "^", 1, true)
	if caretPos then
		local isTetr = sbyte(str, caretPos + 1) == 94
		local rightRaw = str:sub(if isTetr then caretPos + 2 else caretPos + 1)
		if string.find(rightRaw, "^", 1, true) then
			return 0, 1, -1
		end
		local left = trimFast(str:sub(1, caretPos - 1))
		local right = trimFast(rightRaw)
		local lLen = #left
		if lLen == 0 or #right == 0 then
			return 0, 1, -1
		end
		local ex = tonumber(right)
		if ex == nil or ex ~= ex or abs(ex) == huge then
			return 0, 1, -1
		end

		local unaryNeg = false
		local baseStr = left
		local l1 = sbyte(left, 1)
		if l1 == 40 and sbyte(left, lLen) == 41 then
			baseStr = trimFast(left:sub(2, lLen - 1))
		elseif (l1 == 45 or l1 == 43) and lLen >= 3 and sbyte(left, 2) == 40 and sbyte(left, lLen) == 41 then
			unaryNeg = (l1 == 45)
			baseStr = trimFast(left:sub(3, lLen - 1))
		else
			if l1 == 45 then
				unaryNeg = true
				baseStr = left:sub(2)
			elseif l1 == 43 then
				baseStr = left:sub(2)
			end
			if #baseStr == 0 or sbyte(baseStr, 1) <= 32 then
				return 0, 1, -1
			end
		end
		if #baseStr == 0 then
			return 0, 1, -1
		end

		local bm: number, be: number, bl: number
		local bNum = tonumber(baseStr)
		if bNum ~= nil and bNum == bNum and abs(bNum) < huge and bNum ~= 0 then
			bm, be, bl = fromNumberTuple(bNum)
		elseif bNum == 0 and not (string.find(baseStr, "e", 1, true) or string.find(baseStr, "E", 1, true)) then
			bm, be, bl = 0, 0, 0
		elseif not string.find(baseStr, "^", 1, true) then
			bm, be, bl = fromStringTuple(baseStr)
			if bl < 0 then return 0, 1, -1 end
		else
			return 0, 1, -1
		end

		local rm: number, re: number, rl: number
		if isTetr then
			rm, re, rl = tetrTuple(bm, be, bl, ex)
		else
			local em, ee, el = fromNumberTuple(ex)
			rm, re, rl = powTuple(bm, be, bl, em, ee, el)
		end
		if unaryNeg and rl >= 0 then
			rm = -rm
		end
		return rm, re, rl
	end

	local ePos = string.find(str, "e", 1, true) or string.find(str, "E", 1, true)
	if ePos then
		local native = tonumber(str)
		if native ~= nil and native == native and abs(native) < huge and abs(native) >= 1e-300 then
			return fromNumberTuple(native)
		end

		local idx = 1
		local sg = 1
		if b1 == 45 then
			sg = -1
			idx = 2
		elseif b1 == 43 then
			idx = 2
		end

		if idx <= len then
			local c = sbyte(str, idx)
			if c == 101 or c == 69 then
				local chainEnd = idx
				while chainEnd <= len do
					local ch = sbyte(str, chainEnd)
					if ch ~= 101 and ch ~= 69 then break end
					chainEnd += 1
				end
				local tail = str:sub(chainEnd)
				if #tail > 0 and sbyte(tail, 1) > 32 and not string.find(tail, "x", 1, true) and not string.find(tail, "X", 1, true) then
					local val = tonumber(tail)
					if val ~= nil and val == val then
						return fromLegacyTuple(sg, chainEnd - idx, val)
					end
				end
				return 0, 1, -1
			end
		end

		if ePos > 1 then
			local mStr = str:sub(1, ePos - 1)
			local exStr = str:sub(ePos + 1)
			if #mStr > 0 and #exStr > 0 and sbyte(mStr, #mStr) > 32 and sbyte(exStr, 1) > 32 then
				if not string.find(exStr, "e", 1, true) and not string.find(exStr, "E", 1, true) then
					local mantissa = tonumber(mStr)
					local exponent = tonumber(exStr)
					if mantissa ~= nil and exponent ~= nil and mantissa == mantissa and exponent == exponent then
						return fromScientificTuple(mantissa, exponent)
					end
				end
			end
		end
		return 0, 1, -1
	end

	local native = tonumber(str)
	if native ~= nil and native == native and abs(native) < huge then
		return fromNumberTuple(native)
	end

	return 0, 1, -1
end

parseStringTupleForward = fromStringTuple

function ln.fromString(str: string): buffer
	return pack(fromStringTuple(str))
end

function ln.totuple(input: any): (number, number, number)
	return fastUnpack(input)
end
ln.toTuple = ln.totuple
ln.unpack = fastUnpack
ln.pack = pack
ln.pack17 = pack17
ln.write = writeBuf

function ln.fromTuple(s: number, l: number, e: number): buffer
	return pack(fromLegacyTuple(s, l, e))
end

-- Native RemoteEvent, DataStore & JSON plain-table helpers (zero conversion overhead)
function ln.toTable(input: any): { number }
	local m, e, l = fastUnpack(input)
	return { m, e, l }
end

function ln.fromTable(tbl: any): buffer
	if type(tbl) ~= "table" then error("Wrong Type: fromTable(), Input 1") end
	if tbl[1] ~= nil and tbl[2] ~= nil and tbl[3] ~= nil then
		return pack(normalizeTuple(tbl[1], tbl[2], tbl[3]))
	elseif tbl.m ~= nil and tbl.e ~= nil and tbl.l ~= nil then
		return pack(normalizeTuple(tbl.m, tbl.e, tbl.l))
	end
	error("Invalid LightNum table: fromTable()")
end

function ln.tobuffer(input: any): buffer
	local t = type(input)
	if t == "buffer" then
		local len = bufLen(input)
		if len ~= 25 and len ~= 17 then error("Wrong Type: expected 17-byte or 25-byte LightNum buffer") end
		return input
	elseif t == "number" then
		return ln.fromNumber(input)
	elseif t == "string" then
		return ln.fromString(input)
	elseif t == "table" then
		return pack(fastUnpack(input))
	else
		error("Wrong Type: expected LightNum table, buffer, number, string, or wrapper")
	end
end

function ln.tobuffer17(input: any): buffer
	local m, e, l = fastUnpack(input)
	return pack17(m, e, l)
end

function ln.show(n: any): (number, number, number)
	local m, e, l = fastUnpack(n)
	return toLegacyTuple(m, e, l)
end

function ln.showS(n: any): string
	local s17, l17, e17 = ln.show(n)
	return "{" .. s17 .. "," .. l17 .. "," .. e17 .. "}"
end

function ln.showPrecise(n: any): (number, number, number)
	return fastUnpack(n)
end

function ln.showPreciseS(n: any): string
	local m, e, l = fastUnpack(n)
	return "{" .. m .. "," .. e .. "," .. l .. "}"
end

function ln.print(...)
	local args = { ... }
	local out = {}
	for i = 1, #args do
		local arg = args[i]
		local t = type(arg)
		if (t == "buffer" and (bufLen(arg) == 17 or bufLen(arg) == 25)) or (t == "table" and (getmetatable(arg) == WrapMT or arg._buf ~= nil)) then
			out[i] = ln.showS(arg)
		else
			out[i] = tostring(arg)
		end
	end
	print(tconcat(out, " "))
end

function ln.isGammaNum(n: any): boolean
	return type(n) == "buffer" and bufLen(n) == 17
end

function ln.isLightNum(n: any): boolean
	if type(n) ~= "buffer" then return false end
	local len = bufLen(n)
	return len == 25 or len == 17
end

function ln.isWrapped(n: any): boolean
	if type(n) ~= "table" then return false end
	if getmetatable(n) == WrapMT then return true end
	return ln.isLightNum(n._buf)
end

function ln.isTable(n: any): boolean
	return type(n) == "table" and type(n[1]) == "number" and type(n[2]) == "number" and type(n[3]) == "number"
end

function ln.clone(n: any): buffer
	if type(n) == "buffer" then
		local len = bufLen(n)
		if len ~= 25 and len ~= 17 then error("Wrong Type: clone(), Input 1") end
		local out = bufCreate(len)
		bufCopy(out, 0, n, 0, len)
		return out
	end
	return pack(fastUnpack(n))
end

function ln.sign(n: any): number
	local m, _, l = fastUnpack(n)
	if l < 0 then return 1 end
	return if m > 0 then 1 elseif m < 0 then -1 else 0
end

function ln.isPositive(n: any): boolean
	local m, _, l = fastUnpack(n)
	return l >= 0 and m > 0
end

function ln.isNegative(n: any): boolean
	local m, _, l = fastUnpack(n)
	return l >= 0 and m < 0
end

function ln.isZero(n: any): boolean
	local m, _, l = fastUnpack(n)
	return l >= 0 and m == 0
end

function ln.isNaN(n: any): boolean
	local _, _, l = fastUnpack(n)
	return l < 0
end

function ln.isInf(n: any): boolean
	local m, _, l = fastUnpack(n)
	return m ~= 0 and l == huge
end

function ln.isFinite(n: any): boolean
	local m, _, l = fastUnpack(n)
	return l >= 0 and l ~= huge and m == m
end

function ln.isNumber(n: any): boolean
	local m, e, l = fastUnpack(n)
	if l < 0 or l == huge then
		return false
	elseif m == 0 or l == 0 then
		return true
	elseif l == 1 then
		return abs(e + log10(abs(m))) <= INF_LIMIT
	end
	return false
end

local function tupleToNumber(m: number, e: number, l: number): number
	if l < 0 then return 0 / 0 end
	if m == 0 then return 0 end
	if l == huge then return (if m > 0 then 1 else -1) * huge end
	if l == 0 then return m end
	if l == 1 then
		if e > 309 then return (if m > 0 then 1 else -1) * huge end
		if e < -324 then return (if m > 0 then 1 else -1) * 0 end
		return m * (fullPow10[e + 309] or (10 ^ e))
	end
	return if e > 0 then (if m > 0 then 1 else -1) * huge else (if m > 0 then 1 else -1) * 0
end

function ln.toNumber(n: any): number
	if type(n) == "number" then return n end
	return tupleToNumber(fastUnpack(n))
end

-- ============================================================================
-- COMPARISON API
-- ============================================================================

function ln.compare(n1: any, n2: any): number?
	local m1, e1, l1 = fastUnpack(n1)
	local m2, e2, l2 = fastUnpack(n2)
	return compareTuples(m1, e1, l1, m2, e2, l2)
end

function ln.eq(n1: any, n2: any): boolean
	local m1, e1, l1 = fastUnpack(n1)
	local m2, e2, l2 = fastUnpack(n2)
	local cmp = compareTuples(m1, e1, l1, m2, e2, l2)
	return cmp ~= nil and cmp == 0
end

function ln.lt(n1: any, n2: any): boolean
	local m1, e1, l1 = fastUnpack(n1)
	local m2, e2, l2 = fastUnpack(n2)
	local cmp = compareTuples(m1, e1, l1, m2, e2, l2)
	return cmp ~= nil and cmp < 0
end

function ln.lte(n1: any, n2: any): boolean
	local m1, e1, l1 = fastUnpack(n1)
	local m2, e2, l2 = fastUnpack(n2)
	local cmp = compareTuples(m1, e1, l1, m2, e2, l2)
	return cmp ~= nil and cmp <= 0
end

function ln.gt(n1: any, n2: any): boolean
	local m1, e1, l1 = fastUnpack(n1)
	local m2, e2, l2 = fastUnpack(n2)
	local cmp = compareTuples(m1, e1, l1, m2, e2, l2)
	return cmp ~= nil and cmp > 0
end

function ln.gte(n1: any, n2: any): boolean
	local m1, e1, l1 = fastUnpack(n1)
	local m2, e2, l2 = fastUnpack(n2)
	local cmp = compareTuples(m1, e1, l1, m2, e2, l2)
	return cmp ~= nil and cmp >= 0
end

function ln.max(...)
	local count = select("#", ...)
	if count == 0 then error("No Inputs: max()") end
	local bm, be, bl = fastUnpack(select(1, ...))
	for i = 2, count do
		local nm, ne, nl = fastUnpack(select(i, ...))
		local cmp = compareTuples(nm, ne, nl, bm, be, bl)
		if cmp ~= nil and cmp > 0 then
			bm, be, bl = nm, ne, nl
		end
	end
	return pack(bm, be, bl)
end

function ln.min(...)
	local count = select("#", ...)
	if count == 0 then error("No Inputs: min()") end
	local bm, be, bl = fastUnpack(select(1, ...))
	for i = 2, count do
		local nm, ne, nl = fastUnpack(select(i, ...))
		local cmp = compareTuples(nm, ne, nl, bm, be, bl)
		if cmp ~= nil and cmp < 0 then
			bm, be, bl = nm, ne, nl
		end
	end
	return pack(bm, be, bl)
end

function ln.maxabs(...)
	local count = select("#", ...)
	if count == 0 then error("No Inputs: maxabs()") end
	local bm, be, bl = fastUnpack(select(1, ...))
	for i = 2, count do
		local nm, ne, nl = fastUnpack(select(i, ...))
		if bl >= 0 and nl >= 0 then
			if nm ~= 0 and (bm == 0 or compareMagnitude(nm, ne, nl, bm, be, bl) > 0) then
				bm, be, bl = nm, ne, nl
			end
		end
	end
	return pack(bm, be, bl)
end

function ln.minabs(...)
	local count = select("#", ...)
	if count == 0 then error("No Inputs: minabs()") end
	local bm, be, bl = fastUnpack(select(1, ...))
	for i = 2, count do
		local nm, ne, nl = fastUnpack(select(i, ...))
		if bl >= 0 and nl >= 0 then
			if bm ~= 0 and (nm == 0 or compareMagnitude(nm, ne, nl, bm, be, bl) < 0) then
				bm, be, bl = nm, ne, nl
			end
		end
	end
	return pack(bm, be, bl)
end

function ln.maxeq(n1: any, ...)
	local bm, be, bl = fastUnpack(n1)
	local count = select("#", ...)
	for i = 1, count do
		local nm, ne, nl = fastUnpack(select(i, ...))
		local cmp = compareTuples(nm, ne, nl, bm, be, bl)
		if cmp ~= nil and cmp > 0 then
			bm, be, bl = nm, ne, nl
		end
	end
	if type(n1) == "table" and n1[1] ~= nil then
		n1[1], n1[2], n1[3] = bm, be, bl
	else
		writeBuf(fastUnwrapAny(n1), bm, be, bl)
	end
	return n1
end

function ln.mineq(n1: any, ...)
	local bm, be, bl = fastUnpack(n1)
	local count = select("#", ...)
	for i = 1, count do
		local nm, ne, nl = fastUnpack(select(i, ...))
		local cmp = compareTuples(nm, ne, nl, bm, be, bl)
		if cmp ~= nil and cmp < 0 then
			bm, be, bl = nm, ne, nl
		end
	end
	if type(n1) == "table" and n1[1] ~= nil then
		n1[1], n1[2], n1[3] = bm, be, bl
	else
		writeBuf(fastUnwrapAny(n1), bm, be, bl)
	end
	return n1
end

function ln.maxabseq(n1: any, ...)
	local bm, be, bl = fastUnpack(n1)
	local count = select("#", ...)
	for i = 1, count do
		local nm, ne, nl = fastUnpack(select(i, ...))
		if bl >= 0 and nl >= 0 then
			if nm ~= 0 and (bm == 0 or compareMagnitude(nm, ne, nl, bm, be, bl) > 0) then
				bm, be, bl = nm, ne, nl
			end
		end
	end
	if type(n1) == "table" and n1[1] ~= nil then
		n1[1], n1[2], n1[3] = bm, be, bl
	else
		writeBuf(fastUnwrapAny(n1), bm, be, bl)
	end
	return n1
end

function ln.minabseq(n1: any, ...)
	local bm, be, bl = fastUnpack(n1)
	local count = select("#", ...)
	for i = 1, count do
		local nm, ne, nl = fastUnpack(select(i, ...))
		if bl >= 0 and nl >= 0 then
			if bm ~= 0 and (nm == 0 or compareMagnitude(nm, ne, nl, bm, be, bl) < 0) then
				bm, be, bl = nm, ne, nl
			end
		end
	end
	if type(n1) == "table" and n1[1] ~= nil then
		n1[1], n1[2], n1[3] = bm, be, bl
	else
		writeBuf(fastUnwrapAny(n1), bm, be, bl)
	end
	return n1
end

function ln.clamp(n: any, minVal: any, maxVal: any): buffer
	local m, e, l = fastUnpack(n)
	local minM, minE, minL = fastUnpack(minVal)
	local maxM, maxE, maxL = fastUnpack(maxVal)
	if compareTuples(minM, minE, minL, maxM, maxE, maxL) == 1 then
		error("Invalid Range: clamp(), minValue > maxValue")
	end
	if compareTuples(m, e, l, minM, minE, minL) == -1 then
		return pack(minM, minE, minL)
	elseif compareTuples(m, e, l, maxM, maxE, maxL) == 1 then
		return pack(maxM, maxE, maxL)
	end
	return pack(m, e, l)
end

function ln.clampeq(n: any, minVal: any, maxVal: any): any
	local m, e, l = fastUnpack(n)
	local minM, minE, minL = fastUnpack(minVal)
	local maxM, maxE, maxL = fastUnpack(maxVal)
	if compareTuples(minM, minE, minL, maxM, maxE, maxL) == 1 then
		error("Invalid Range: clampeq(), minValue > maxValue")
	end
	if compareTuples(m, e, l, minM, minE, minL) == -1 then
		m, e, l = minM, minE, minL
	elseif compareTuples(m, e, l, maxM, maxE, maxL) == 1 then
		m, e, l = maxM, maxE, maxL
	end
	if type(n) == "table" and n[1] ~= nil then
		n[1], n[2], n[3] = m, e, l
	else
		writeBuf(fastUnwrapAny(n), m, e, l)
	end
	return n
end

-- ============================================================================
-- PURE ARITHMETIC & HYPER-OPERATION API
-- ============================================================================

-- Common boxed-buffer fast decoder. The public 25-byte ABI stores l/e/m at
-- fixed offsets; bypass the generic any-type and 17/25-byte dispatch when all
-- operands are already native LightNum buffers. Legacy buffers and all other
-- accepted input types continue through fastUnpack unchanged.
local function fastUnpackPair(a: any, b: any): (number, number, number, number, number, number)
	local ta, tb = type(a), type(b)
	if ta == "buffer" and tb == "buffer" and bufLen(a) == 25 and bufLen(b) == 25 then
		return bufReadF64(a, 17), bufReadF64(a, 9), bufReadF64(a, 1),
			bufReadF64(b, 17), bufReadF64(b, 9), bufReadF64(b, 1)
	end
	if tb == "number" then
		local m1, e1, l1
		if ta == "buffer" and bufLen(a) == 25 then
			m1, e1, l1 = bufReadF64(a, 17), bufReadF64(a, 9), bufReadF64(a, 1)
		else
			m1, e1, l1 = fastUnpack(a)
		end
		local ab = abs(b)
		if b == b and ab > 1e-10 and ab < 1e10 then
			return m1, e1, l1, b, 0, 0
		end
		local m2, e2, l2 = fromNumberTuple(b)
		return m1, e1, l1, m2, e2, l2
	elseif ta == "number" then
		local m2, e2, l2
		if tb == "buffer" and bufLen(b) == 25 then
			m2, e2, l2 = bufReadF64(b, 17), bufReadF64(b, 9), bufReadF64(b, 1)
		else
			m2, e2, l2 = fastUnpack(b)
		end
		local aa = abs(a)
		if a == a and aa > 1e-10 and aa < 1e10 then
			return a, 0, 0, m2, e2, l2
		end
		local m1, e1, l1 = fromNumberTuple(a)
		return m1, e1, l1, m2, e2, l2
	end
	local m1, e1, l1 = fastUnpack(a)
	local m2, e2, l2 = fastUnpack(b)
	return m1, e1, l1, m2, e2, l2
end

local function fastUnpackTriple(a: any, b: any, c: any): (number, number, number, number, number, number, number, number, number)
	if type(a) == "buffer" and type(b) == "buffer" and type(c) == "buffer"
		and bufLen(a) == 25 and bufLen(b) == 25 and bufLen(c) == 25 then
		return bufReadF64(a, 17), bufReadF64(a, 9), bufReadF64(a, 1),
			bufReadF64(b, 17), bufReadF64(b, 9), bufReadF64(b, 1),
			bufReadF64(c, 17), bufReadF64(c, 9), bufReadF64(c, 1)
	end
	local m1, e1, l1 = fastUnpack(a)
	local m2, e2, l2 = fastUnpack(b)
	local m3, e3, l3 = fastUnpack(c)
	return m1, e1, l1, m2, e2, l2, m3, e3, l3
end

function ln.add(n1: any, n2: any): buffer
	local m1, e1, l1, m2, e2, l2 = fastUnpackPair(n1, n2)
	return pack(addTuple(m1, e1, l1, m2, e2, l2))
end

function ln.sub(n1: any, n2: any): buffer
	local m1, e1, l1, m2, e2, l2 = fastUnpackPair(n1, n2)
	return pack(subTuple(m1, e1, l1, m2, e2, l2))
end

function ln.mul(n1: any, n2: any): buffer
	local m1, e1, l1, m2, e2, l2 = fastUnpackPair(n1, n2)
	return pack(mulTuple(m1, e1, l1, m2, e2, l2))
end

function ln.div(n1: any, n2: any): buffer
	local m1, e1, l1, m2, e2, l2 = fastUnpackPair(n1, n2)
	return pack(divTuple(m1, e1, l1, m2, e2, l2))
end

-- Optional destination-buffer variants. The caller owns a writable 25-byte
-- buffer; these avoid allocating a new result buffer on repeated hot loops.
local function validateDestination25(outBuf: any, functionName: string): buffer
	if type(outBuf) ~= "buffer" or bufLen(outBuf) ~= 25 then
		error("Wrong Type: " .. functionName .. " expects a 25-byte destination buffer", 2)
	end
	return outBuf
end

function ln.addInto(outBuf: buffer, n1: any, n2: any): buffer
	local out = validateDestination25(outBuf, "addInto()")
	local m1,e1,l1,m2,e2,l2 = fastUnpackPair(n1,n2)
	local rm,re,rl = addTuple(m1,e1,l1,m2,e2,l2)
	return writeBuf(out,rm,re,rl)
end

function ln.mulInto(outBuf: buffer, n1: any, n2: any): buffer
	local out = validateDestination25(outBuf, "mulInto()")
	local m1,e1,l1,m2,e2,l2 = fastUnpackPair(n1,n2)
	local rm,re,rl = mulTuple(m1,e1,l1,m2,e2,l2)
	return writeBuf(out,rm,re,rl)
end

function ln.fma(a: any, b: any, c: any): buffer
	local m1, e1, l1, m2, e2, l2, m3, e3, l3 = fastUnpackTriple(a, b, c)
	return pack(fmaTuple(m1, e1, l1, m2, e2, l2, m3, e3, l3))
end

function ln.intdiv(n1: any, n2: any): buffer
	local m1, e1, l1 = fastUnpack(n1)
	local m2, e2, l2 = fastUnpack(n2)
	return pack(intdivTuple(m1, e1, l1, m2, e2, l2))
end

function ln.mod(n1: any, n2: any): buffer
	local m1, e1, l1 = fastUnpack(n1)
	local m2, e2, l2 = fastUnpack(n2)
	return pack(modTuple(m1, e1, l1, m2, e2, l2))
end

function ln.recip(n: any): buffer
	local m, e, l = fastUnpack(n)
	return pack(recipTuple(m, e, l))
end

function ln.abs(n: any): buffer
	local m, e, l = fastUnpack(n)
	return pack(abs(m), e, l)
end

function ln.neg(n: any): buffer
	local m, e, l = fastUnpack(n)
	if l < 0 then return pack(0, 1, -1) end
	return pack(-m, e, l)
end

function ln.pow(n1: any, n2: any): buffer
	local m1, e1, l1 = fastUnpack(n1)
	local m2, e2, l2 = fastUnpack(n2)
	return pack(powTuple(m1, e1, l1, m2, e2, l2))
end

function ln.root(n1: any, n2: any): buffer
	local m1, e1, l1 = fastUnpack(n1)
	local m2, e2, l2 = fastUnpack(n2)
	return pack(rootTuple(m1, e1, l1, m2, e2, l2))
end

function ln.sqrt(n: any): buffer
	local m, e, l = fastUnpack(n)
	return pack(sqrtTuple(m, e, l))
end

function ln.cbrt(n: any): buffer
	local m, e, l = fastUnpack(n)
	return pack(cbrtTuple(m, e, l))
end

function ln.pow2(n: any): buffer
	local m, e, l = fastUnpack(n)
	return pack(pow2Tuple(m, e, l))
end

function ln.pow10(n: any): buffer
	local m, e, l = fastUnpack(n)
	return pack(pow10Tuple(m, e, l))
end
ln.exp10 = ln.pow10

function ln.exp(n: any): buffer
	local m, e, l = fastUnpack(n)
	return pack(expTuple(m, e, l))
end

function ln.log(n1: any, n2: any): buffer
	local m1, e1, l1 = fastUnpack(n1)
	local m2, e2, l2 = fastUnpack(n2)
	return pack(logTuple(m1, e1, l1, m2, e2, l2))
end

function ln.log10(n: any): buffer
	local m, e, l = fastUnpack(n)
	return pack(log10Tuple(m, e, l))
end

function ln.abslog10(n: any): buffer
	local m, e, l = fastUnpack(n)
	return pack(abslog10Tuple(m, e, l))
end

function ln.log2(n: any): buffer
	local m, e, l = fastUnpack(n)
	return pack(log2Tuple(m, e, l))
end

function ln.ln(n: any): buffer
	local m, e, l = fastUnpack(n)
	return pack(lnTuple(m, e, l))
end

function ln.tetr(n1: any, n2: any): buffer
	local m1, e1, l1 = fastUnpack(n1)
	local m2, e2, l2 = fastUnpack(n2)
	if l2 < 0 then return pack(0, 1, -1) end
	if l2 == huge then return pack(tetrTuple(m1, e1, l1, (if m2 > 0 then 1 else -1) * huge)) end
	local height: number
	if m2 == 0 then
		height = 0
	elseif l2 == 0 then
		height = m2
	elseif l2 == 1 and e2 <= 308 then
		height = m2 * (10 ^ e2)
	else
		return pack(0, 1, -1)
	end
	return pack(tetrTuple(m1, e1, l1, height))
end

function ln.slog(n: any, base: any?): number
	local m1, e1, l1 = fastUnpack(n)
	local bm, be, bl
	if base == nil then
		bm, be, bl = 10, 0, 0
	else
		bm, be, bl = fastUnpack(base)
	end
	return slogTuple(m1, e1, l1, bm, be, bl)
end

-- ============================================================================
-- ROUNDING API
-- ============================================================================

function ln.floor(n: any): buffer
	return pack(floorTuple(fastUnpack(n)))
end

function ln.ceil(n: any): buffer
	return pack(ceilTuple(fastUnpack(n)))
end

function ln.round(n: any): buffer
	return pack(roundTuple(fastUnpack(n)))
end

function ln.roundto(n1: any, n2: any): buffer
	local m1, e1, l1 = fastUnpack(n1)
	local m2, e2, l2 = fastUnpack(n2)
	if m2 == 0 or l1 < 0 or l2 < 0 then
		return pack(0, 1, -1)
	end
	local dm, de, dl = divTuple(m1, e1, l1, m2, e2, l2)
	local rm, re, rl = roundTuple(dm, de, dl)
	return pack(mulTuple(rm, re, rl, m2, e2, l2))
end

-- ============================================================================
-- ZERO-ALLOCATION MUTABLE (*eq) API (SUPPORTS TABLES, 25B & 17B BUFFERS)
-- ============================================================================

local function commitMutation(target: any, rm: number, re: number, rl: number): any
	if type(target) == "table" and target[1] ~= nil then
		target[1], target[2], target[3] = rm, re, rl
		return target
	end
	writeBuf(fastUnwrapAny(target), rm, re, rl)
	return target
end

function ln.set(n: any, s: any, l: number?, e: number?): any
	if l == nil and e == nil and type(s) ~= "number" then
		local rm, re, rl = fastUnpack(s)
		return commitMutation(n, rm, re, rl)
	end
	if type(s) ~= "number" then error("Wrong Type: set(), Input 2") end
	local layer = if l == nil then 0 else l
	local expVal = if e == nil then 0 else e
	if type(layer) ~= "number" then error("Wrong Type: set(), Input 3") end
	if type(expVal) ~= "number" then error("Wrong Type: set(), Input 4") end
	local rm, re, rl = fromLegacyTuple(s, layer, expVal)
	return commitMutation(n, rm, re, rl)
end

function ln.setFromBuffer(n: any, source: any): any
	local m, e, l = fastUnpack(source)
	return commitMutation(n, m, e, l)
end

function ln.setFromNumber(n: any, number: number): any
	if type(number) ~= "number" then error("Wrong Type: setFromNumber(), Input 2") end
	local rm, re, rl = fromNumberTuple(number)
	return commitMutation(n, rm, re, rl)
end

function ln.writeTuple(n: any, s: number, l: number, e: number): any
	local rm, re, rl = fromLegacyTuple(s, l, e)
	return commitMutation(n, rm, re, rl)
end

function ln.copy(n1: any, n2: any): any
	local m, e, l = fastUnpack(n2)
	return commitMutation(n1, m, e, l)
end

local function commitTableSlots(tbl: { number }, rm: number, re: number, rl: number)
	if rm ~= tbl[1] then tbl[1] = rm end
	if re ~= tbl[2] then tbl[2] = re end
	if rl ~= tbl[3] then tbl[3] = rl end
	return tbl
end

function ln.addeq(n1: any, n2: any): any
	local m2, e2, l2 = fastUnpack(n2)
	if type(n1) == "table" and n1[1] ~= nil then
		local m1, e1, l1 = n1[1], n1[2], n1[3]
		if l1 == 0 and l2 == 0 then
			local n = m1 + m2
			if n == 0 then
				if m1 ~= 0 then n1[1] = 0 end
				return n1
			end
			local an = if n > 0 then n else -n
			if an < 1e10 and an > 1e-10 then
				n1[1] = n
				return n1
			end
		elseif l1 == 1 and l2 == 1 then
			if e1 > e2 or (e1 == e2 and abs(m1) >= abs(m2)) then
				local diff = e1 - e2
				if diff >= 17 then return n1 end
				if diff == 0 then
					local m = m1 + m2
					local am = if m > 0 then m else -m
					if am >= 1 and am < 10 then
						n1[1] = m
						return n1
					elseif am >= 10 and e1 ~= 9 then
						n1[1] = m * 0.1
						n1[2] = e1 + 1
						return n1
					end
				else
					local m = m1 + m2 * (negPow10[diff] or (10 ^ (-diff)))
					local am = if m > 0 then m else -m
					if am >= 1 and am < 10 and e1 ~= -10 then
						n1[1] = m
						return n1
					end
				end
			elseif e2 - e1 >= 17 then
				n1[1] = m2
				n1[2] = e2
				return n1
			end
		end
		return commitTableSlots(n1, addTuple(m1, e1, l1, m2, e2, l2))
	end
	local b1 = fastUnwrapAny(n1)
	local m1, e1, l1 = readAnyBuffer(b1)
	writeBuf(b1, addTuple(m1, e1, l1, m2, e2, l2))
	return n1
end

function ln.subeq(n1: any, n2: any): any
	local m2, e2, l2 = fastUnpack(n2)
	if type(n1) == "table" and n1[1] ~= nil then
		local m1, e1, l1 = n1[1], n1[2], n1[3]
		if l1 == 0 and l2 == 0 then
			local n = m1 - m2
			if n == 0 then
				if m1 ~= 0 then n1[1] = 0 end
				return n1
			end
			local an = if n > 0 then n else -n
			if an < 1e10 and an > 1e-10 then
				n1[1] = n
				return n1
			end
		elseif l1 == 1 and l2 == 1 then
			local sm2 = -m2
			if e1 > e2 or (e1 == e2 and abs(m1) >= abs(sm2)) then
				local diff = e1 - e2
				if diff >= 17 then return n1 end
				if diff == 0 then
					local m = m1 + sm2
					local am = if m > 0 then m else -m
					if am >= 1 and am < 10 then
						n1[1] = m
						return n1
					elseif am >= 10 and e1 ~= 9 then
						n1[1] = m * 0.1
						n1[2] = e1 + 1
						return n1
					end
				else
					local m = m1 + sm2 * (negPow10[diff] or (10 ^ (-diff)))
					local am = if m > 0 then m else -m
					if am >= 1 and am < 10 and e1 ~= -10 then
						n1[1] = m
						return n1
					end
				end
			elseif e2 - e1 >= 17 then
				n1[1] = sm2
				n1[2] = e2
				return n1
			end
		end
		return commitTableSlots(n1, subTuple(m1, e1, l1, m2, e2, l2))
	end
	local b1 = fastUnwrapAny(n1)
	local m1, e1, l1 = readAnyBuffer(b1)
	writeBuf(b1, subTuple(m1, e1, l1, m2, e2, l2))
	return n1
end

function ln.muleq(n1: any, n2: any): any
	local m2, e2, l2 = fastUnpack(n2)
	if type(n1) == "table" and n1[1] ~= nil then
		local m1, e1, l1 = n1[1], n1[2], n1[3]
		if l1 == 0 and l2 == 0 then
			local n = m1 * m2
			if n == 0 then
				if m1 ~= 0 then n1[1] = 0 end
				return n1
			end
			local an = if n > 0 then n else -n
			if an < 1e10 and an > 1e-10 then
				n1[1] = n
				return n1
			end
		elseif l1 == 1 and l2 == 0 then
			local m = m1 * m2
			local am = if m > 0 then m else -m
			if am >= 1 and am < 10 and e1 ~= -10 then
				n1[1] = m
				return n1
			elseif am >= 10 and am < 100 then
				local e = e1 + 1
				if e >= 10 or e < -10 then
					n1[1] = m * 0.1
					n1[2] = e
					return n1
				end
			end
		elseif l1 == 1 and l2 == 1 then
			local m = m1 * m2
			local e = e1 + e2
			if abs(m) >= 10 then
				m *= 0.1
				e += 1
			end
			if (e >= 10 or e < -10) and abs(e) < huge then
				n1[1] = m
				n1[2] = e
				return n1
			end
		end
		return commitTableSlots(n1, mulTuple(m1, e1, l1, m2, e2, l2))
	end
	local b1 = fastUnwrapAny(n1)
	local m1, e1, l1 = readAnyBuffer(b1)
	writeBuf(b1, mulTuple(m1, e1, l1, m2, e2, l2))
	return n1
end

function ln.diveq(n1: any, n2: any): any
	local m2, e2, l2 = fastUnpack(n2)
	if type(n1) == "table" and n1[1] ~= nil then
		local m1, e1, l1 = n1[1], n1[2], n1[3]
		if l1 == 0 and l2 == 0 and m2 ~= 0 then
			local n = m1 / m2
			if n == 0 then
				if m1 ~= 0 then n1[1] = 0 end
				return n1
			end
			local an = if n > 0 then n else -n
			if an < 1e10 and an > 1e-10 then
				n1[1] = n
				return n1
			end
		elseif l1 == 1 and l2 == 1 then
			local m = m1 / m2
			local e = e1 - e2
			if abs(m) < 1 then
				m *= 10
				e -= 1
			end
			if (e >= 10 or e < -10) and abs(e) < huge then
				n1[1] = m
				n1[2] = e
				return n1
			end
		end
		return commitTableSlots(n1, divTuple(m1, e1, l1, m2, e2, l2))
	end
	local b1 = fastUnwrapAny(n1)
	local m1, e1, l1 = readAnyBuffer(b1)
	writeBuf(b1, divTuple(m1, e1, l1, m2, e2, l2))
	return n1
end

function ln.fmaeq(n1: any, a: any, b: any): any
	local ma, ea, la = fastUnpack(a)
	local mb, eb, lb = fastUnpack(b)
	if type(n1) == "table" and n1[1] ~= nil then
		return commitTableSlots(n1, fmaTuple(ma, ea, la, mb, eb, lb, n1[1], n1[2], n1[3]))
	end
	local b1 = fastUnwrapAny(n1)
	local m1, e1, l1 = readAnyBuffer(b1)
	writeBuf(b1, fmaTuple(ma, ea, la, mb, eb, lb, m1, e1, l1))
	return n1
end

function ln.intdiveq(n1: any, n2: any): any
	local m2, e2, l2 = fastUnpack(n2)
	if type(n1) == "table" and n1[1] ~= nil then
		n1[1], n1[2], n1[3] = intdivTuple(n1[1], n1[2], n1[3], m2, e2, l2)
		return n1
	end
	local b1 = fastUnwrapAny(n1)
	local m1, e1, l1 = readAnyBuffer(b1)
	writeBuf(b1, intdivTuple(m1, e1, l1, m2, e2, l2))
	return n1
end

function ln.modeq(n1: any, n2: any): any
	local m2, e2, l2 = fastUnpack(n2)
	if type(n1) == "table" and n1[1] ~= nil then
		n1[1], n1[2], n1[3] = modTuple(n1[1], n1[2], n1[3], m2, e2, l2)
		return n1
	end
	local b1 = fastUnwrapAny(n1)
	local m1, e1, l1 = readAnyBuffer(b1)
	writeBuf(b1, modTuple(m1, e1, l1, m2, e2, l2))
	return n1
end

function ln.recipeq(n: any): any
	local m, e, l = fastUnpack(n)
	return commitMutation(n, recipTuple(m, e, l))
end

function ln.abseq(n: any): any
	if type(n) == "table" and n[1] ~= nil then
		if n[1] < 0 then n[1] = -n[1] end
		return n
	end
	local m, e, l = fastUnpack(n)
	if m < 0 then m = -m end
	return commitMutation(n, m, e, l)
end

function ln.negeq(n: any): any
	if type(n) == "table" and n[1] ~= nil then
		if n[3] >= 0 then n[1] = -n[1] end
		return n
	end
	local m, e, l = fastUnpack(n)
	if l >= 0 then m = -m end
	return commitMutation(n, m, e, l)
end

function ln.poweq(n1: any, n2: any): any
	local m2, e2, l2 = fastUnpack(n2)
	if type(n1) == "table" and n1[1] ~= nil then
		n1[1], n1[2], n1[3] = powTuple(n1[1], n1[2], n1[3], m2, e2, l2)
		return n1
	end
	local b1 = fastUnwrapAny(n1)
	local m1, e1, l1 = readAnyBuffer(b1)
	writeBuf(b1, powTuple(m1, e1, l1, m2, e2, l2))
	return n1
end

function ln.rooteq(n1: any, n2: any): any
	local m2, e2, l2 = fastUnpack(n2)
	if type(n1) == "table" and n1[1] ~= nil then
		n1[1], n1[2], n1[3] = rootTuple(n1[1], n1[2], n1[3], m2, e2, l2)
		return n1
	end
	local b1 = fastUnwrapAny(n1)
	local m1, e1, l1 = readAnyBuffer(b1)
	writeBuf(b1, rootTuple(m1, e1, l1, m2, e2, l2))
	return n1
end

function ln.sqrteq(n: any): any
	local m, e, l = fastUnpack(n)
	return commitMutation(n, sqrtTuple(m, e, l))
end

function ln.cbrteq(n: any): any
	local m, e, l = fastUnpack(n)
	return commitMutation(n, cbrtTuple(m, e, l))
end

function ln.pow2eq(n: any): any
	local m, e, l = fastUnpack(n)
	return commitMutation(n, pow2Tuple(m, e, l))
end

function ln.pow10eq(n: any): any
	local m, e, l = fastUnpack(n)
	return commitMutation(n, pow10Tuple(m, e, l))
end

function ln.expeq(n: any): any
	local m, e, l = fastUnpack(n)
	return commitMutation(n, expTuple(m, e, l))
end

function ln.logeq(n1: any, n2: any): any
	local m2, e2, l2 = fastUnpack(n2)
	if type(n1) == "table" and n1[1] ~= nil then
		n1[1], n1[2], n1[3] = logTuple(n1[1], n1[2], n1[3], m2, e2, l2)
		return n1
	end
	local b1 = fastUnwrapAny(n1)
	local m1, e1, l1 = readAnyBuffer(b1)
	writeBuf(b1, logTuple(m1, e1, l1, m2, e2, l2))
	return n1
end

function ln.log10eq(n: any): any
	local m, e, l = fastUnpack(n)
	return commitMutation(n, log10Tuple(m, e, l))
end

function ln.abslog10eq(n: any): any
	local m, e, l = fastUnpack(n)
	return commitMutation(n, abslog10Tuple(m, e, l))
end

function ln.log2eq(n: any): any
	local m, e, l = fastUnpack(n)
	return commitMutation(n, log2Tuple(m, e, l))
end

function ln.lneq(n: any): any
	local m, e, l = fastUnpack(n)
	return commitMutation(n, lnTuple(m, e, l))
end

function ln.tetreq(n1: any, n2: any): any
	local m2, e2, l2 = fastUnpack(n2)
	if l2 < 0 then return commitMutation(n1, 0, 1, -1) end
	local height: number
	if l2 == huge then
		height = (if m2 > 0 then 1 else -1) * huge
	elseif m2 == 0 then
		height = 0
	elseif l2 == 0 then
		height = m2
	elseif l2 == 1 and e2 <= 308 then
		height = m2 * (10 ^ e2)
	else
		return commitMutation(n1, 0, 1, -1)
	end
	local m1, e1, l1 = fastUnpack(n1)
	return commitMutation(n1, tetrTuple(m1, e1, l1, height))
end

function ln.flooreq(n: any): any
	local m, e, l = fastUnpack(n)
	return commitMutation(n, floorTuple(m, e, l))
end

function ln.ceileq(n: any): any
	local m, e, l = fastUnpack(n)
	return commitMutation(n, ceilTuple(m, e, l))
end

function ln.roundeq(n: any): any
	local m, e, l = fastUnpack(n)
	return commitMutation(n, roundTuple(m, e, l))
end

function ln.roundtoeq(n1: any, n2: any): any
	local m1, e1, l1 = fastUnpack(n1)
	local m2, e2, l2 = fastUnpack(n2)
	if m2 == 0 or l1 < 0 or l2 < 0 then
		return commitMutation(n1, 0, 1, -1)
	end
	local dm, de, dl = divTuple(m1, e1, l1, m2, e2, l2)
	local rm, re, rl = roundTuple(dm, de, dl)
	return commitMutation(n1, mulTuple(rm, re, rl, m2, e2, l2))
end

-- ============================================================================
-- BATCH ACCUMULATORS, ECONOMY MATH & INTERPOLATION
-- ============================================================================

function ln.sum(...): buffer
	local count = select("#", ...)
	if count == 0 then return bufCreate(25) end
	local m, e, l = fastUnpack(select(1, ...))
	for i = 2, count do
		local nm, ne, nl = fastUnpack(select(i, ...))
		m, e, l = addTuple(m, e, l, nm, ne, nl)
	end
	return pack(m, e, l)
end

function ln.prod(...): buffer
	local count = select("#", ...)
	if count == 0 then return pack(1, 0, 0) end
	local m, e, l = fastUnpack(select(1, ...))
	for i = 2, count do
		local nm, ne, nl = fastUnpack(select(i, ...))
		m, e, l = mulTuple(m, e, l, nm, ne, nl)
	end
	return pack(m, e, l)
end

function ln.sumArray(arr: { any }): buffer
	local n = #arr
	if n == 0 then return bufCreate(25) end
	local m, e, l = fastUnpack(arr[1])
	for i = 2, n do
		local item = arr[i]
		local it = type(item)
		local nm, ne, nl
		if it == "table" and item[1] ~= nil then
			nm, ne, nl = item[1], item[2], item[3]
		elseif it == "number" then
			nm, ne, nl = fromNumberTuple(item)
		elseif it == "buffer" and bufLen(item) == 25 then
			nm, ne, nl = bufReadF64(item, 17), bufReadF64(item, 9), bufReadF64(item, 1)
		else
			nm, ne, nl = fastUnpack(item)
		end
		m, e, l = addTuple(m, e, l, nm, ne, nl)
	end
	return pack(m, e, l)
end

function ln.prodArray(arr: { any }): buffer
	local n = #arr
	if n == 0 then return pack(1, 0, 0) end
	local m, e, l = fastUnpack(arr[1])
	for i = 2, n do
		local item = arr[i]
		local it = type(item)
		local nm, ne, nl
		if it == "table" and item[1] ~= nil then
			nm, ne, nl = item[1], item[2], item[3]
		elseif it == "number" then
			nm, ne, nl = fromNumberTuple(item)
		elseif it == "buffer" and bufLen(item) == 25 then
			nm, ne, nl = bufReadF64(item, 17), bufReadF64(item, 9), bufReadF64(item, 1)
		else
			nm, ne, nl = fastUnpack(item)
		end
		m, e, l = mulTuple(m, e, l, nm, ne, nl)
	end
	return pack(m, e, l)
end

function ln.lerp(a: any, b: any, t: any): buffer
	local am, ae, al = fastUnpack(a)
	local bm, be, bl = fastUnpack(b)
	local tm, te, tl = fastUnpack(t)
	local dm, de, dl = subTuple(bm, be, bl, am, ae, al)
	local pm, pe, pl = mulTuple(dm, de, dl, tm, te, tl)
	return pack(addTuple(am, ae, al, pm, pe, pl))
end

function ln.lerpeq(a: any, b: any, t: any): any
	local am, ae, al = fastUnpack(a)
	local bm, be, bl = fastUnpack(b)
	local tm, te, tl = fastUnpack(t)
	local dm, de, dl = subTuple(bm, be, bl, am, ae, al)
	local pm, pe, pl = mulTuple(dm, de, dl, tm, te, tl)
	return commitMutation(a, addTuple(am, ae, al, pm, pe, pl))
end

function ln.invLerp(a: any, b: any, v: any): number
	local am, ae, al = fastUnpack(a)
	local bm, be, bl = fastUnpack(b)
	local vm, ve, vl = fastUnpack(v)
	if am == bm and ae == be and al == bl then return 0 end
	local nm, ne, nl = subTuple(vm, ve, vl, am, ae, al)
	local dm, de, dl = subTuple(bm, be, bl, am, ae, al)
	return tupleToNumber(divTuple(nm, ne, nl, dm, de, dl))
end

function ln.logLerp(a: any, b: any, t: any): buffer
	local am, ae, al = fastUnpack(a)
	local bm, be, bl = fastUnpack(b)
	local tm, te, tl = fastUnpack(t)
	if am <= 0 or bm <= 0 then return pack(0, 1, -1) end
	local rm, re, rl = divTuple(bm, be, bl, am, ae, al)
	local pm, pe, pl = powTuple(rm, re, rl, tm, te, tl)
	return pack(mulTuple(am, ae, al, pm, pe, pl))
end

function ln.hypot(a: any, b: any): buffer
	local am, ae, al = fastUnpack(a)
	local bm, be, bl = fastUnpack(b)
	if al < 0 or bl < 0 then return pack(0, 1, -1) end
	if al == huge or bl == huge then return pack(1, 1, huge) end
	if am == 0 then return pack(abs(bm), be, bl) end
	if bm == 0 then return pack(abs(am), ae, al) end
	if al == 0 and bl == 0 and abs(am) < 1e5 and abs(bm) < 1e5 then
		return pack(math.sqrt(am * am + bm * bm), 0, 0)
	end
	local a2m, a2e, a2l = powTuple(abs(am), ae, al, 2, 0, 0)
	local b2m, b2e, b2l = powTuple(abs(bm), be, bl, 2, 0, 0)
	local sm, se, sl = addTuple(a2m, a2e, a2l, b2m, b2e, b2l)
	return pack(sqrtTuple(sm, se, sl))
end

function ln.random(n1: any, n2: any): buffer
	local m1, e1, l1 = fastUnpack(n1)
	local m2, e2, l2 = fastUnpack(n2)
	if l1 < 0 or l2 < 0 or l1 == huge or l2 == huge then return pack(0, 1, -1) end
	local dm, de, dl = subTuple(m2, e2, l2, m1, e1, l1)
	local rm, re, rl = fromNumberTuple(mrandom())
	local pm, pe, pl = mulTuple(dm, de, dl, rm, re, rl)
	return pack(addTuple(pm, pe, pl, m1, e1, l1))
end

function ln.randomeq(n1: any, n2: any): any
	local m1, e1, l1 = fastUnpack(n1)
	local m2, e2, l2 = fastUnpack(n2)
	if l1 < 0 or l2 < 0 or l1 == huge or l2 == huge then
		return commitMutation(n1, 0, 1, -1)
	end
	local dm, de, dl = subTuple(m2, e2, l2, m1, e1, l1)
	local rm, re, rl = fromNumberTuple(mrandom())
	local pm, pe, pl = mulTuple(dm, de, dl, rm, re, rl)
	return commitMutation(n1, addTuple(pm, pe, pl, m1, e1, l1))
end

function ln.geosum(base: any, mul: any, start: any, last: any): buffer
	local bm, be, bl = fastUnpack(base)
	local mm, me, ml = fastUnpack(mul)
	local sm, se, sl = fastUnpack(start)
	local lm, le, ll = fastUnpack(last)
	return pack(geosumTuple(bm, be, bl, mm, me, ml, sm, se, sl, lm, le, ll))
end

function ln.geosumeq(base: any, mul: any, start: any, last: any): any
	local bm, be, bl = fastUnpack(base)
	local mm, me, ml = fastUnpack(mul)
	local sm, se, sl = fastUnpack(start)
	local lm, le, ll = fastUnpack(last)
	return commitMutation(base, geosumTuple(bm, be, bl, mm, me, ml, sm, se, sl, lm, le, ll))
end

function ln.geosumR(base: any, mul: any, start: any, amount: any): buffer
	local bm, be, bl = fastUnpack(base)
	local mm, me, ml = fastUnpack(mul)
	local sm, se, sl = fastUnpack(start)
	local amm, ame, aml = fastUnpack(amount)
	return pack(geosumRTuple(bm, be, bl, mm, me, ml, sm, se, sl, amm, ame, aml))
end

function ln.geosumReq(base: any, mul: any, start: any, amount: any): any
	local bm, be, bl = fastUnpack(base)
	local mm, me, ml = fastUnpack(mul)
	local sm, se, sl = fastUnpack(start)
	local amm, ame, aml = fastUnpack(amount)
	return commitMutation(base, geosumRTuple(bm, be, bl, mm, me, ml, sm, se, sl, amm, ame, aml))
end

function ln.costBulk(baseCost: any, ratio: any, owned: any, count: any): buffer
	local bm, be, bl = fastUnpack(baseCost)
	local rm, re, rl = fastUnpack(ratio)
	local om, oe, ol = fastUnpack(owned)
	local cm, ce, cl = fastUnpack(count)
	if cm <= 0 or cl < 0 then return bufCreate(25) end
	local l1m, l1e, l1l = addTuple(om, oe, ol, cm, ce, cl)
	local lastM, lastE, lastL = subTuple(l1m, l1e, l1l, 1, 0, 0)
	return pack(geosumTuple(bm, be, bl, rm, re, rl, om, oe, ol, lastM, lastE, lastL))
end

function ln.maxBuy(currency: any, baseCost: any, ratio: any, owned: any): buffer
	local curM, curE, curL = fastUnpack(currency)
	local bm, be, bl = fastUnpack(baseCost)
	local rm, re, rl = fastUnpack(ratio)
	local om, oe, ol = fastUnpack(owned)
	if curM <= 0 or curL < 0 or bm <= 0 or bl < 0 or ol < 0 or rl < 0 then
		return bufCreate(25)
	end
	if ol == huge then return bufCreate(25) end
	if curL == huge then return pack(1, 1, huge) end
	local prM, prE, prL = powTuple(rm, re, rl, om, oe, ol)
	local firstM, firstE, firstL = mulTuple(bm, be, bl, prM, prE, prL)
	if compareTuples(curM, curE, curL, firstM, firstE, firstL) == -1 then
		return bufCreate(25)
	end
	local endM, endE, endL = geosumRTuple(bm, be, bl, rm, re, rl, om, oe, ol, curM, curE, curL)
	if endL < 0 then return bufCreate(25) end
	if endL == huge then return pack(1, 1, huge) end
	local diffM, diffE, diffL = subTuple(endM, endE, endL, om, oe, ol)
	local cntM, cntE, cntL = addTuple(diffM, diffE, diffL, 1, 0, 0)
	cntM, cntE, cntL = floorTuple(cntM, cntE, cntL)
	if cntM <= 0 or cntL < 0 then return bufCreate(25) end

	if cntL == 0 or (cntL == 1 and cntE <= 15) then
		local nextM, nextE, nextL = addTuple(cntM, cntE, cntL, 1, 0, 0)
		local nEndM, nEndE, nEndL = addTuple(om, oe, ol, cntM, cntE, cntL)
		local cNextM, cNextE, cNextL = geosumTuple(bm, be, bl, rm, re, rl, om, oe, ol, nEndM, nEndE, nEndL)
		local cmpNext = compareTuples(cNextM, cNextE, cNextL, curM, curE, curL)
		if cmpNext ~= nil and cmpNext <= 0 then
			return pack(nextM, nextE, nextL)
		end
	end
	return pack(cntM, cntE, cntL)
end

function ln.gamma(n: any): buffer
	return pack(gammaTuple(fastUnpack(n)))
end

function ln.gammaeq(n: any): any
	return commitMutation(n, gammaTuple(fastUnpack(n)))
end

function ln.fact(n: any): buffer
	local m, e, l = fastUnpack(n)
	local am, ae, al = addTuple(m, e, l, 1, 0, 0)
	return pack(gammaTuple(am, ae, al))
end

function ln.facteq(n: any): any
	local m, e, l = fastUnpack(n)
	local am, ae, al = addTuple(m, e, l, 1, 0, 0)
	return commitMutation(n, gammaTuple(am, ae, al))
end

-- ============================================================================
-- FORMATTING & NOTATION API (53-BIT EXACT AT LAYER 1 + LEGACY SUPPORT)
-- ============================================================================

local suffixes = {
	{
		{ "k", "M", "B", "T", "Qd", "Qt", "Sx", "Sp", "Oc", "No" },
		{ "", "U", "D", "T", "Qt", "Qn", "Sx", "Sp", "Oc", "No" },
		{ "", "Dc", "Vt", "Tg", "qg", "Qg", "sg", "Sg", "og", "ng" },
		{ "", "Ce", "Du", "Tr", "Qa", "Qi", "Se", "Si", "Ot", "Ni" },
	},
}

function ln.suffix(i: number): string
	if type(i) ~= "number" or i ~= i then error("Wrong Type: suffix(), Input 1") end
	if i <= 0 then return "" end
	if i == huge then return "Mi" end
	i = floor(i)
	if i <= 10 then return suffixes[1][1][i] end
	if i >= 1001 then return "Mi" end
	return suffixes[1][2][((i - 1) % 10) + 1]
		.. suffixes[1][3][(((i - 1) // 10) % 10) + 1]
		.. suffixes[1][4][(((i - 1) // 100) % 10) + 1]
end

local function addcommas(n: number): string
	if n ~= n then return "NaN" end
	if abs(n) == huge then return if n < 0 then "-Inf" else "Inf" end
	local sg = if n < 0 then "-" else ""
	local digits = tostring(floor(abs(n)))
	local first = #digits % 3
	if first == 0 then first = 3 end
	local parts = tcreate((#digits - first) // 3 + 1)
	parts[1] = string.sub(digits, 1, first)
	local part = 2
	for i = first + 1, #digits, 3 do
		parts[part] = string.sub(digits, i, i + 2)
		part += 1
	end
	return sg .. tconcat(parts, ",")
end

function ln.std(s: number, l: number, e: number): string
	if l < 0 then return "NaN" end
	if l == huge or abs(e) == huge then return if s < 0 then "-Inf" else "Inf" end
	if s == 0 or l >= 2 or (l == 1 and e >= INF_LIMIT) then return "0" end
	local n = s * (if l == 0 then e else 10 ^ e)
	if abs(n) >= 1 then
		local exponent = floor(log10(abs(n)))
		local mod3 = exponent % 3
		return sformat(digitcuts[mmax(ln.DefaultTotalDigits - mod3, 0)], n * pow10Table[mod3 - exponent])
	else
		return sformat(digitcuts[mmax(ln.DefaultTotalDigits, 0)], s * e)
	end
end

function ln.stdsetround(s: number, l: number, e: number, i: number): string
	if l < 0 then return "NaN" end
	if l == huge or abs(e) == huge then return if s < 0 then "-Inf" else "Inf" end
	if s == 0 or l >= 2 or (l == 1 and e >= INF_LIMIT) then return "0" end
	local n = s * (if l == 0 then e else 10 ^ e)
	if abs(n) >= 1 then
		local exponent = floor(log10(abs(n)))
		local mod3 = exponent % 3
		return sformat(digitcuts[mmax(i - mod3, 0)], n * pow10Table[mod3 - exponent])
	else
		return sformat(digitcuts[mmax(i, 0)], s * e)
	end
end

function ln.suf(s: number, l: number, e: number): string
	if l < 0 then return "NaN" end
	if l == huge or abs(e) == huge then return if s < 0 then "-Inf" else "Inf" end
	if s == 0 then
		return "0"
	elseif l == 0 then
		if e >= 1e-3 and e < 1e3 then
			local mod3 = if e < 10 then 0 elseif e < 100 then 1 else 2
			local rp = mmax(ln.DefaultTotalDigits, mod3)
			local a = pow10Table[mod3]
			return sformat(digitcuts[rp - mod3], s * (if e > a * capthres[rp] then a * capvalue[rp] else e))
		end
		local exp, man = modf(abs(log10(e)))
		local mod3 = exp % 3
		local rp = mmax(ln.DefaultTotalDigits, mod3)
		return (if s < 0 then "-" else "")
			.. (if e < 1 then "1/" else "")
			.. sformat(digitcuts[rp - mod3], pow10Table[mod3] * (if man > capthresl10[rp] then capvalue[rp] else 10 ^ man))
			.. ln.suffix((exp - mod3) / 3)
	elseif l == 1 and abs(e) < 3006 then
		local esign = sign(e)
		local exp, man = modf(abs(e))
		local mod3 = exp % 3
		local rp = mmax(ln.DefaultTotalDigits, mod3)
		return (if s < 0 then "-" else "")
			.. (if esign < 0 then "1/" else "")
			.. sformat(digitcuts[rp - mod3], pow10Table[mod3] * (if man > capthresl10[rp] then capvalue[rp] else 10 ^ man))
			.. ln.suffix((exp - mod3) / 3)
	elseif l <= ln.MaxEs or (l == ln.MaxEs + 1 and abs(e) < 3006) then
		local esign = sign(e)
		e = abs(e)
		local exp, man = modf(if e < 3006 then e else log10(e))
		local mod3 = exp % 3
		local rp = mmax(ln.DefaultTotalDigits, mod3)
		return (if s < 0 then "-" else "")
			.. (if esign < 0 then "1/" else "1")
			.. Etrail[if e < 3006 then l - 1 else l]
			.. sformat(digitcuts[rp - mod3], pow10Table[mod3] * (if man > capthresl10[rp] then capvalue[rp] else 10 ^ man))
			.. ln.suffix((exp - mod3) / 3)
	elseif l < 1e6 then
		local expmant = log10(abs(e))
		return sformat(digitcuts[ln.DefaultDigits], s * (if expmant > capthres[ln.DefaultDigits] then capvalue[ln.DefaultDigits] else expmant))
			.. "L" .. (if e > 0 then "+" else "-") .. addcommas(l)
	else
		local exp, man = modf(log10(l))
		local mod3 = exp % 3
		local rp = mmax(ln.DefaultTotalDigits, mod3)
		local expmant = log10(abs(e))
		return sformat(digitcuts[ln.DefaultDigits], s * (if expmant > capthres[ln.DefaultDigits] then capvalue[ln.DefaultDigits] else expmant))
			.. "L" .. (if e > 0 then "+" else "-")
			.. sformat(digitcuts[rp - mod3], (if man > capthresl10[rp] then capvalue[rp] else 10 ^ man) * pow10Table[mod3])
			.. ln.suffix((exp - mod3) / 3)
	end
end

function ln.sci(s: number, l: number, e: number): string
	if l < 0 then return "NaN" end
	if l == huge or abs(e) == huge then return if s < 0 then "-Inf" else "Inf" end
	if s == 0 then
		return "0e0"
	elseif l == 0 then
		local exp = floor(log10(e))
		local man = e / pow10Table[exp]
		return sformat(digitcuts[ln.DefaultDigits], s * (if man > capthres[ln.DefaultDigits] then capvalue[ln.DefaultDigits] else man)) .. "e" .. exp
	elseif l == 1 and abs(e) < 1e6 then
		local esign = sign(e)
		e = abs(e)
		local exp, man = modf(e)
		return sformat(digitcuts[ln.DefaultDigits], s * (if man > capthresl10[ln.DefaultDigits] then capvalue[ln.DefaultDigits] else 10 ^ man)) .. "e" .. addcommas(esign * exp)
	elseif l <= ln.MaxEs or (l == ln.MaxEs + 1 and abs(e) < 1e6) then
		local esign = sign(e)
		e = abs(e)
		local exp, man = modf(if e < 1e6 then e else log10(e))
		return (if s < 0 then "-" else "")
			.. Etrail[if e < 1e6 then l - 1 else l]
			.. sformat(digitcuts[ln.DefaultDigits], esign * (if man > capthresl10[ln.DefaultDigits] then capvalue[ln.DefaultDigits] else 10 ^ man))
			.. "e" .. addcommas(exp)
	elseif l < 1e6 then
		local expmant = log10(abs(e))
		return sformat(digitcuts[ln.DefaultDigits], s * (if expmant > capthres[ln.DefaultDigits] then capvalue[ln.DefaultDigits] else expmant))
			.. "L" .. (if e > 0 then "+" else "-") .. addcommas(l)
	else
		local exp, man = modf(log10(l))
		local expmant = log10(abs(e))
		return sformat(digitcuts[ln.DefaultDigits], s * (if expmant > capthres[ln.DefaultDigits] then capvalue[ln.DefaultDigits] else expmant))
			.. "L" .. (if e > 0 then "+" else "-")
			.. sformat(digitcuts[ln.DefaultDigits], (if man > capthresl10[ln.DefaultDigits] then capvalue[ln.DefaultDigits] else 10 ^ man))
			.. "e" .. exp
	end
end

function ln.eng(s: number, l: number, e: number): string
	if l < 0 then return "NaN" end
	if l == huge or abs(e) == huge then return if s < 0 then "-Inf" else "Inf" end
	if s == 0 then
		return "0e0"
	elseif l == 0 then
		local exp, man = modf(log10(e))
		if exp < 0 then
			exp -= 1
			man += 1
		end
		local mod3 = exp % 3
		local rp = mmax(ln.DefaultTotalDigits, mod3)
		return (if s < 0 then "-" else "")
			.. sformat(digitcuts[rp - mod3], pow10Table[mod3] * (if man > capthresl10[rp] then capvalue[rp] else 10 ^ man))
			.. "e" .. (exp - mod3)
	elseif l == 1 and abs(e) < 1000002 then
		local esign = sign(e)
		e = abs(e)
		local exp, man = modf(e)
		local mod3 = exp % 3
		local rp = mmax(ln.DefaultTotalDigits, mod3)
		return sformat(digitcuts[rp - mod3], s * pow10Table[mod3] * (if man > capthresl10[rp] then capvalue[rp] else 10 ^ man))
			.. "e" .. addcommas(esign * (exp - mod3))
	elseif l <= ln.MaxEs or (l == ln.MaxEs + 1 and abs(e) < 1000002) then
		local esign = sign(e)
		e = abs(e)
		local exp, man = modf(if e < 1000002 then e else log10(e))
		local mod3 = exp % 3
		local rp = mmax(ln.DefaultTotalDigits, mod3)
		return (if s < 0 then "-" else "")
			.. Etrail[if e < 1000002 then l - 1 else l]
			.. sformat(digitcuts[rp - mod3], pow10Table[mod3] * (if man > capthresl10[rp] then capvalue[rp] else 10 ^ man))
			.. "e" .. addcommas(esign * (exp - mod3))
	elseif l < 1e6 then
		local expmant = log10(abs(e))
		return sformat(digitcuts[ln.DefaultDigits], s * (if expmant > capthres[ln.DefaultDigits] then capvalue[ln.DefaultDigits] else expmant))
			.. "L" .. (if e > 0 then "+" else "-") .. addcommas(l)
	else
		local exp, man = modf(log10(l))
		local mod3 = exp % 3
		local rp = mmax(ln.DefaultTotalDigits, mod3)
		local expmant = log10(abs(e))
		return sformat(digitcuts[ln.DefaultDigits], s * (if expmant > capthres[ln.DefaultDigits] then capvalue[ln.DefaultDigits] else expmant))
			.. "L" .. (if e > 0 then "+" else "-")
			.. sformat(digitcuts[rp - mod3], pow10Table[mod3] * (if man > capthresl10[rp] then capvalue[rp] else 10 ^ man))
			.. "e" .. (exp - mod3)
	end
end

function ln.lay(s: number, l: number, e: number): string
	if l < 0 then return "NaN" end
	if l == huge or abs(e) == huge then return if s < 0 then "-Inf" else "Inf" end
	if s == 0 then
		return "0L+0"
	elseif l == 0 then
		return sformat(digitcuts[ln.DefaultDigits], s * abs(log10(e)))
			.. "L" .. (if e >= 1 then "+" else "-") .. "0"
	elseif l < 1e6 then
		local expmant = log10(abs(e))
		return sformat(digitcuts[ln.DefaultDigits], s * (if expmant > capthres[ln.DefaultDigits] then capvalue[ln.DefaultDigits] else expmant))
			.. "L" .. (if e > 0 then "+" else "-") .. addcommas(l)
	else
		local exp, man = modf(log10(l))
		local expmant = log10(abs(e))
		return sformat(digitcuts[ln.DefaultDigits], s * (if expmant > capthres[ln.DefaultDigits] then capvalue[ln.DefaultDigits] else expmant))
			.. "L" .. (if e > 0 then "+" else "-")
			.. sformat(digitcuts[ln.DefaultDigits], (if man > capthresl10[ln.DefaultDigits] then capvalue[ln.DefaultDigits] else 10 ^ man))
			.. "e" .. exp
	end
end

local function sufPrecise(m: number, e: number, l: number): string
	if l == 1 and abs(e) < 3006 then
		local s = if m > 0 then 1 else -1
		local am = abs(m)
		if e >= 0 then
			local mod3 = e % 3
			local rp = mmax(ln.DefaultTotalDigits, mod3)
			local val = (if am > capthres[rp] then capvalue[rp] else am) * pow10Table[mod3]
			return (if s < 0 then "-" else "") .. sformat(digitcuts[rp - mod3], val) .. ln.suffix((e - mod3) / 3)
		else
			local re = if am == 1 then -e else -e - 1
			local rm = if am == 1 then 1 else 10 / am
			local mod3 = re % 3
			local rp = mmax(ln.DefaultTotalDigits, mod3)
			local val = (if rm > capthres[rp] then capvalue[rp] else rm) * pow10Table[mod3]
			return (if s < 0 then "-" else "") .. "1/" .. sformat(digitcuts[rp - mod3], val) .. ln.suffix((re - mod3) / 3)
		end
	end
	local s17, l17, e17 = toLegacyTuple(m, e, l)
	return ln.suf(s17, l17, e17)
end

local function sciPrecise(m: number, e: number, l: number): string
	if l == 1 and abs(e) < 1e6 then
		local s = if m > 0 then 1 else -1
		local am = abs(m)
		local capped = if am > capthres[ln.DefaultDigits] then capvalue[ln.DefaultDigits] else am
		return sformat(digitcuts[ln.DefaultDigits], s * capped) .. "e" .. addcommas(e)
	end
	local s17, l17, e17 = toLegacyTuple(m, e, l)
	return ln.sci(s17, l17, e17)
end

local function engPrecise(m: number, e: number, l: number): string
	if l == 1 and abs(e) < 1000002 then
		local s = if m > 0 then 1 else -1
		local am = abs(m)
		local mod3 = e % 3
		local rp = mmax(ln.DefaultTotalDigits, mod3)
		local capped = if am > capthres[rp] then capvalue[rp] else am
		return sformat(digitcuts[rp - mod3], s * pow10Table[mod3] * capped) .. "e" .. addcommas(e - mod3)
	end
	local s17, l17, e17 = toLegacyTuple(m, e, l)
	return ln.eng(s17, l17, e17)
end

function ln.toSuffix(n: any): string
	local m, e, l = fastUnpack(n)
	return sufPrecise(m, e, l)
end

function ln.toScientific(n: any): string
	local m, e, l = fastUnpack(n)
	return sciPrecise(m, e, l)
end

function ln.toEngineer(n: any): string
	local m, e, l = fastUnpack(n)
	return engPrecise(m, e, l)
end

function ln.toLayered(n: any): string
	local m, e, l = fastUnpack(n)
	local s17, l17, e17 = toLegacyTuple(m, e, l)
	return ln.lay(s17, l17, e17)
end

function ln.toHyper(n: any): string
	local m, e, l = fastUnpack(n)
	if l < 0 then return "NaN" end
	if l == huge then return if m < 0 then "-Inf" else "Inf" end
	if m == 0 then return "0" end
	if l <= 1 and abs(e) < 1e10 then return sufPrecise(m, e, l) end
	if e < 0 then
		local _, l17, e17 = toLegacyTuple(m, e, l)
		return (if m < 0 then "-1/" else "1/") .. "10^^" .. tostring(l17) .. "^" .. sformat("%.3f", abs(e17))
	end
	local prefix = if m < 0 then "-" else ""
	local frac = slogTuple(abs(m), e, l, 10, 0, 0)
	return prefix .. "10^^" .. sformat("%.4f", frac)
end

function ln.tostring(n: any, suffixtype: number?, intround: number?): string
	local m, e, l = fastUnpack(n)
	local st = if suffixtype == nil then ln.DefaultSuffixType else suffixtype
	if st == ln.SuffixTypes.Scientific then
		return sciPrecise(m, e, l)
	end
	-- Only suffix modes that inspect the legacy tuple pay this conversion cost.
	local s17, l17, e17 = toLegacyTuple(m, e, l)
	if st == ln.SuffixTypes.Scientific3 then
		return if l17 >= 1 or e17 >= 1e3 or e17 <= 1e-3 then sciPrecise(m, e, l) else (if intround then ln.stdsetround(s17, l17, e17, intround) else ln.std(s17, l17, e17))
	elseif st == ln.SuffixTypes.Scientific36 then
		return if l17 >= 2 or (l17 == 1 and (e17 >= 36 or e17 < 0)) or (l17 == 0 and e17 <= 1e-3) then sciPrecise(m, e, l) elseif (l17 == 1 and e17 > 0) or (l17 == 0 and e17 >= 1e3) then sufPrecise(m, e, l) else (if intround then ln.stdsetround(s17, l17, e17, intround) else ln.std(s17, l17, e17))
	elseif st == ln.SuffixTypes.Scientific306 then
		return if l17 >= 2 or (l17 == 1 and (e17 >= 306 or e17 < 0)) or (l17 == 0 and e17 <= 1e-3) then sciPrecise(m, e, l) elseif (l17 == 1 and e17 > 0) or (l17 == 0 and e17 >= 1e3) then sufPrecise(m, e, l) else (if intround then ln.stdsetround(s17, l17, e17, intround) else ln.std(s17, l17, e17))
	elseif st == ln.SuffixTypes.Scientific3006 then
		return if l17 >= 2 or (l17 == 1 and (e17 >= 3006 or e17 < 0)) or (l17 == 0 and e17 <= 1e-3) then sciPrecise(m, e, l) elseif (l17 == 1 and e17 > 0) or (l17 == 0 and e17 >= 1e3) then sufPrecise(m, e, l) else (if intround then ln.stdsetround(s17, l17, e17, intround) else ln.std(s17, l17, e17))
	elseif st == ln.SuffixTypes.Engineer then
		return engPrecise(m, e, l)
	elseif st == ln.SuffixTypes.Engineer3 then
		return if l17 >= 1 or e17 >= 1e3 or e17 <= 1e-3 then engPrecise(m, e, l) else (if intround then ln.stdsetround(s17, l17, e17, intround) else ln.std(s17, l17, e17))
	elseif st == ln.SuffixTypes.Engineer36 then
		return if l17 >= 2 or (l17 == 1 and (e17 >= 36 or e17 < 0)) or (l17 == 0 and e17 <= 1e-3) then engPrecise(m, e, l) elseif (l17 == 1 and e17 > 0) or (l17 == 0 and e17 >= 1e3) then sufPrecise(m, e, l) else (if intround then ln.stdsetround(s17, l17, e17, intround) else ln.std(s17, l17, e17))
	elseif st == ln.SuffixTypes.Engineer306 then
		return if l17 >= 2 or (l17 == 1 and (e17 >= 306 or e17 < 0)) or (l17 == 0 and e17 <= 1e-3) then engPrecise(m, e, l) elseif (l17 == 1 and e17 > 0) or (l17 == 0 and e17 >= 1e3) then sufPrecise(m, e, l) else (if intround then ln.stdsetround(s17, l17, e17, intround) else ln.std(s17, l17, e17))
	elseif st == ln.SuffixTypes.Engineer3006 then
		return if l17 >= 2 or (l17 == 1 and (e17 >= 3006 or e17 < 0)) or (l17 == 0 and e17 <= 1e-3) then engPrecise(m, e, l) elseif (l17 == 1 and e17 > 0) or (l17 == 0 and e17 >= 1e3) then sufPrecise(m, e, l) else (if intround then ln.stdsetround(s17, l17, e17, intround) else ln.std(s17, l17, e17))
	elseif st == ln.SuffixTypes.Layered then
		return ln.lay(s17, l17, e17)
	end
	error("Unknown SuffixType: tostring(), Input 2")
end

-- ============================================================================
-- DATASTORE CODECS (100% 17-BYTE GAMMANUM WIRE-COMPATIBLE)
-- ============================================================================

function ln.lbencode(n: any): number
	if type(n) == "number" and (n ~= n or abs(n) == huge) then
		error("Cannot lbencode NaN or infinity")
	end
	local s, l, e = ln.show(n)
	if l < 0 or l == huge then
		error("Cannot lbencode NaN or infinity")
	end
	if s == 0 then
		return 0
	elseif l == 0 then
		return s * round(4503599627370496 + 2199023255552 * (log10(e) / 10))
	elseif l >= 1024 and e > 0 then
		return s * (6755399441055744 + 2199023255552 * mlog(l + (log10(e) - 1) / 9, 2))
	elseif l < 1024 and e > 0 then
		return s * round(4503599627370496 + 2199023255552 * (l + (log10(e) - 1) / 9))
	elseif l < 1024 and e < 0 then
		return s * round(2251799813685248 + 2199023255552 * (1024 - l - (log10(-e) - 1) / 9))
	else
		return s * (2199023255552 * (1024 - mlog(l + (log10(-e) - 1) / 9, 2)))
	end
end

function ln.lbdecode(n: number): buffer
	if type(n) ~= "number" then error("Wrong Type: lbdecode(), Input 1") end
	if n ~= n or abs(n) == huge then error("Invalid non-finite score: lbdecode()") end
	if n == 0 then return bufCreate(17) end
	local s = if n > 0 then 1 else -1
	n = abs(n)
	if abs(n - 4503599627370496) <= 2199023255552 then
		return ln.createCheckless(s, 0, 10 ^ ((n - 4503599627370496) / 219902325555.2))
	elseif n <= 2251799813685248 then
		n = 2 ^ (1024 - (n / 2199023255552))
		local fl = floor(n)
		return ln.createCheckless(s, fl, -(10 ^ (1 + (n - fl) * 9)))
	elseif n <= 4503599627370496 then
		n = 1024 - ((n - 2251799813685248) / 2199023255552)
		local fl = floor(n)
		return ln.createCheckless(s, fl, -(10 ^ (1 + (n - fl) * 9)))
	elseif n <= 6755399441055744 then
		n = (n - 4503599627370496) / 2199023255552
		local fl = floor(n)
		return ln.createCheckless(s, fl, 10 ^ (1 + (n - fl) * 9))
	else
		n = 2 ^ ((n - 6755399441055744) / 2199023255552)
		local fl = floor(n)
		return ln.createCheckless(s, fl, 10 ^ (1 + (n - fl) * 9))
	end
end

local B64_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
local B64_CHARS = table.create(64)
local B64_BYTE_VAL = {}
for i = 1, #B64_ALPHABET do
	local ch = B64_ALPHABET:sub(i, i)
	B64_CHARS[i - 1] = ch
	B64_BYTE_VAL[sbyte(ch, 1)] = i - 1
end

function ln.b64encode(n: any): string
	local input = ln.tobuffer17(n)
	local out = table.create(24)
	local idx = 1
	for i = 0, 16, 3 do
		local b1 = bufReadU8(input, i)
		local has2 = i + 1 < 17
		local has3 = i + 2 < 17
		local b2 = if has2 then bufReadU8(input, i + 1) else 0
		local b3 = if has3 then bufReadU8(input, i + 2) else 0
		local packed = b1 * 65536 + b2 * 256 + b3
		out[idx] = B64_CHARS[(packed // 262144) % 64]
		out[idx + 1] = B64_CHARS[(packed // 4096) % 64]
		idx += 2
		if has2 then
			out[idx] = B64_CHARS[(packed // 64) % 64]
			idx += 1
		end
		if has3 then
			out[idx] = B64_CHARS[packed % 64]
			idx += 1
		end
	end
	return tconcat(out)
end

function ln.b64decode(str: string): buffer
	if type(str) ~= "string" then error("Wrong Type: b64decode(), Input 1") end
	str = str:gsub("%s+", ""):gsub("=+$", "")
	local len = #str
	if len % 4 == 1 then error("Invalid Base64: b64decode()") end
	local bytes = table.create(17)
	local outIdx = 1
	local i = 1
	while i <= len do
		local c1 = B64_BYTE_VAL[sbyte(str, i)]
		local c2 = B64_BYTE_VAL[sbyte(str, i + 1)]
		local c3 = if i + 2 <= len then B64_BYTE_VAL[sbyte(str, i + 2)] else nil
		local c4 = if i + 3 <= len then B64_BYTE_VAL[sbyte(str, i + 3)] else nil
		if c1 == nil or c2 == nil or (i + 2 <= len and c3 == nil) or (i + 3 <= len and c4 == nil) then
			error("Invalid Base64: b64decode()")
		end
		local packed = c1 * 262144 + c2 * 4096 + (c3 or 0) * 64 + (c4 or 0)
		bytes[outIdx] = schar((packed // 65536) % 256)
		outIdx += 1
		if c3 ~= nil then
			bytes[outIdx] = schar((packed // 256) % 256)
			outIdx += 1
		end
		if c4 ~= nil then
			bytes[outIdx] = schar(packed % 256)
			outIdx += 1
		end
		i += 4
	end
	local raw = tconcat(bytes)
	if #raw ~= 17 then error("Invalid LightNum Base64 length: b64decode()") end
	return bufFromString(raw)
end

function ln.hexencode(n: any): string
	local input = ln.tobuffer17(n)
	local chars = table.create(17)
	for i = 0, 16 do
		local a = bufReadU8(input, i)
		local n1 = a // 16
		local n2 = a % 16
		chars[i + 1] = schar(if n1 < 10 then n1 + 48 else n1 + 87, if n2 < 10 then n2 + 48 else n2 + 87)
	end
	return tconcat(chars)
end

local function hexVal(byte: number): number?
	if byte >= 48 and byte <= 57 then return byte - 48 end
	if byte >= 65 and byte <= 70 then return byte - 55 end
	if byte >= 97 and byte <= 102 then return byte - 87 end
	return nil
end

function ln.hexdecode(str: string): buffer
	if type(str) ~= "string" then error("Wrong Type: hexdecode(), Input 1") end
	if #str ~= 34 then error("Invalid LightNum hex length: hexdecode()") end
	local chars = table.create(17)
	for i = 1, 34, 2 do
		local hi = hexVal(sbyte(str, i))
		local lo = hexVal(sbyte(str, i + 1))
		if hi == nil or lo == nil then error("Invalid Hex: hexdecode()") end
		chars[(i + 1) // 2] = schar(hi * 16 + lo)
	end
	return bufFromString(tconcat(chars))
end

-- ============================================================================
-- ZERO-ALLOCATION PUBLIC REGISTER-TUPLE API (v1.20 (m, e, l) KERNELS)
-- ============================================================================

local function negTuple(m: number, e: number, l: number): (number, number, number)
	if l < 0 then return 0, 1, -1 end
	return -m, e, l
end

local function absTuple(m: number, e: number, l: number): (number, number, number)
	return (if m < 0 then -m else m), e, l
end

local function factTuple(m: number, e: number, l: number): (number, number, number)
	local am, ae, al = addTuple(m, e, l, 1, 0, 0)
	return gammaTuple(am, ae, al)
end

local function eqTuple(m1: number, e1: number, l1: number, m2: number, e2: number, l2: number): boolean
	if l1 == 0 and l2 == 0 then
		return m1 == m2
	elseif l1 == 1 and l2 == 1 then
		return m1 == m2 and e1 == e2
	end
	local cmp = compareTuples(m1, e1, l1, m2, e2, l2)
	return cmp ~= nil and cmp == 0
end

local function ltTuple(m1: number, e1: number, l1: number, m2: number, e2: number, l2: number): boolean
	if l1 == 0 and l2 == 0 then
		return m1 < m2
	elseif l1 == 1 and l2 == 1 then
		if m1 > 0 and m2 <= 0 then return false end
		if m1 < 0 and m2 >= 0 then return true end
		if m1 > 0 then
			return e1 < e2 or (e1 == e2 and m1 < m2)
		else
			return e1 > e2 or (e1 == e2 and m1 < m2)
		end
	end
	local cmp = compareTuples(m1, e1, l1, m2, e2, l2)
	return cmp ~= nil and cmp < 0
end

local function lteTuple(m1: number, e1: number, l1: number, m2: number, e2: number, l2: number): boolean
	if l1 == 0 and l2 == 0 then
		return m1 <= m2
	elseif l1 == 1 and l2 == 1 then
		if m1 > 0 and m2 <= 0 then return false end
		if m1 < 0 and m2 >= 0 then return true end
		if m1 > 0 then
			return e1 < e2 or (e1 == e2 and m1 <= m2)
		else
			return e1 > e2 or (e1 == e2 and m1 <= m2)
		end
	end
	local cmp = compareTuples(m1, e1, l1, m2, e2, l2)
	return cmp ~= nil and cmp <= 0
end

local function gtTuple(m1: number, e1: number, l1: number, m2: number, e2: number, l2: number): boolean
	if l1 == 0 and l2 == 0 then
		return m1 > m2
	elseif l1 == 1 and l2 == 1 then
		if m1 > 0 and m2 <= 0 then return true end
		if m1 < 0 and m2 >= 0 then return false end
		if m1 > 0 then
			return e1 > e2 or (e1 == e2 and m1 > m2)
		else
			return e1 < e2 or (e1 == e2 and m1 > m2)
		end
	end
	local cmp = compareTuples(m1, e1, l1, m2, e2, l2)
	return cmp ~= nil and cmp > 0
end

local function gteTuple(m1: number, e1: number, l1: number, m2: number, e2: number, l2: number): boolean
	if l1 == 0 and l2 == 0 then
		return m1 >= m2
	elseif l1 == 1 and l2 == 1 then
		if m1 > 0 and m2 <= 0 then return true end
		if m1 < 0 and m2 >= 0 then return false end
		if m1 > 0 then
			return e1 > e2 or (e1 == e2 and m1 >= m2)
		else
			return e1 < e2 or (e1 == e2 and m1 >= m2)
		end
	end
	local cmp = compareTuples(m1, e1, l1, m2, e2, l2)
	return cmp ~= nil and cmp >= 0
end

local function toScientificTuple(m: number, e: number, l: number): (number, number)
	if l < 0 then return 0 / 0, 0 end
	if m == 0 then return 0, 0 end
	if l == huge then return (if m > 0 then 1 else -1) * huge, 0 end
	if l == 0 then
		local exp = floor(log10(abs(m)))
		local man = m / (10 ^ exp)
		if abs(man) >= 10 then
			man *= 0.1
			exp += 1
		elseif abs(man) < 1 then
			man *= 10
			exp -= 1
		end
		return man, exp
	elseif l == 1 then
		return m, e
	end
	return (if e > 0 then (if m > 0 then 1 else -1) * huge else 0), (if e > 0 then huge else -huge)
end

-- Opt-in Luau native f32 SIMD vector interop (~7 decimal digits; core/SoA stay 53-bit f64)
local vecCreate = (vector and vector.create) or (Vector3 and Vector3.new)
local function tupleToVector(m: number, e: number, l: number): any
	if vecCreate then
		return vecCreate(m, e, l)
	end
	return { x = m, y = e, z = l, X = m, Y = e, Z = l }
end

local function fromVectorTuple(v: any): (number, number, number)
	local m = if v.x ~= nil then v.x else v.X
	local e = if v.y ~= nil then v.y else v.Y
	local l = if v.z ~= nil then v.z else v.Z
	return normalizeTuple(m, e, l)
end

ln.Tuple = {
	normalize = normalizeTuple,
	fromNumber = fromNumberTuple,
	fromScientific = fromScientificTuple,
	fromString = fromStringTuple,
	fromBuffer = readAnyBuffer,
	toBuffer = pack,
	toBuffer17 = pack17,
	writeBuffer = writeBuf,
	unpack = fastUnpack,
	toNumber = tupleToNumber,
	toScientific = toScientificTuple,
	toVector = tupleToVector,
	fromVector = fromVectorTuple,
	add = addTuple,
	sub = subTuple,
	mul = mulTuple,
	div = divTuple,
	fma = fmaTuple,
	intdiv = intdivTuple,
	mod = modTuple,
	neg = negTuple,
	abs = absTuple,
	recip = recipTuple,
	pow = powTuple,
	root = rootTuple,
	sqrt = sqrtTuple,
	cbrt = cbrtTuple,
	pow2 = pow2Tuple,
	pow10 = pow10Tuple,
	exp = expTuple,
	log = logTuple,
	log10 = log10Tuple,
	abslog10 = abslog10Tuple,
	log2 = log2Tuple,
	ln = lnTuple,
	tetr = tetrTuple,
	slog = slogTuple,
	floor = floorTuple,
	ceil = ceilTuple,
	round = roundTuple,
	gamma = gammaTuple,
	fact = factTuple,
	geosum = geosumTuple,
	geosumR = geosumRTuple,
	compare = compareTuples,
	eq = eqTuple,
	lt = ltTuple,
	lte = lteTuple,
	gt = gtTuple,
	gte = gteTuple,
}

ln.addT = addTuple
ln.subT = subTuple
ln.mulT = mulTuple
ln.divT = divTuple
ln.fmaT = fmaTuple
ln.powT = powTuple
ln.sqrtT = sqrtTuple
ln.log10T = log10Tuple
ln.lnT = lnTuple
ln.expT = expTuple
ln.eqT = eqTuple
ln.ltT = ltTuple
ln.lteT = lteTuple
ln.gtT = gtTuple
ln.gteT = gteTuple

function ln.toVector(n: any): any
	return tupleToVector(fastUnpack(n))
end

function ln.fromVector(v: any): buffer
	return pack(fromVectorTuple(v))
end

-- ============================================================================
-- STRUCTURE-OF-ARRAYS (SoA) 53-BIT BATCH ENGINE (ZERO GC BARRIERS)
-- ============================================================================

export type LightNumSoA = {
	length: number,
	capacity: number,
	m: buffer,
	e: buffer,
	l: buffer,
}

local SoA = {}

function SoA.new(count: number): LightNumSoA
	local n = if count > 0 then floor(count) else 0
	return {
		length = n,
		capacity = n,
		m = bufCreate(n * 8),
		e = bufCreate(n * 8),
		l = bufCreate(n * 8),
	}
end

function SoA.fromArray(arr: { any }): LightNumSoA
	local n = #arr
	local soa = SoA.new(n)
	local mBuf, eBuf, lBuf = soa.m, soa.e, soa.l
	for i = 1, n do
		local m, e, l = fastUnpack(arr[i])
		local off = (i - 1) * 8
		bufWriteF64(mBuf, off, m)
		bufWriteF64(eBuf, off, e)
		bufWriteF64(lBuf, off, l)
	end
	return soa
end

function SoA.getTuple(soa: LightNumSoA, idx: number): (number, number, number)
	local off = (idx - 1) * 8
	return bufReadF64(soa.m, off), bufReadF64(soa.e, off), bufReadF64(soa.l, off)
end

function SoA.getBuffer(soa: LightNumSoA, idx: number): buffer
	local off = (idx - 1) * 8
	return pack(bufReadF64(soa.m, off), bufReadF64(soa.e, off), bufReadF64(soa.l, off))
end

function SoA.setTuple(soa: LightNumSoA, idx: number, m: number, e: number, l: number): LightNumSoA
	local off = (idx - 1) * 8
	bufWriteF64(soa.m, off, m)
	bufWriteF64(soa.e, off, e)
	bufWriteF64(soa.l, off, l)
	return soa
end

function SoA.set(soa: LightNumSoA, idx: number, val: any): LightNumSoA
	local m, e, l = fastUnpack(val)
	local off = (idx - 1) * 8
	bufWriteF64(soa.m, off, m)
	bufWriteF64(soa.e, off, e)
	bufWriteF64(soa.l, off, l)
	return soa
end

function SoA.addScalar(soa: LightNumSoA, scalar: any): LightNumSoA
	local m2, e2, l2 = fastUnpack(scalar)
	-- Canonical zero is an exact identity for every valid LightNum tuple.
	if m2 == 0 and e2 == 0 and l2 == 0 then return soa end
	local n = soa.length
	if n == 0 then return soa end
	local mBuf, eBuf, lBuf = soa.m, soa.e, soa.l
	local limit = (n - 1) * 8
	for off = 0, limit, 8 do
		local m1 = bufReadF64(mBuf, off)
		local e1 = bufReadF64(eBuf, off)
		local l1 = bufReadF64(lBuf, off)
		local rm, re, rl
		if l1 == 0 and l2 == 0 then
			-- Exact inline equivalent of addTuple's Layer-0 + Layer-0 branch.
			local nsum = m1 + m2
			if nsum == 0 then
				rm, re, rl = 0, 0, 0
			else
				local an = if nsum > 0 then nsum else -nsum
				if an < 1e10 and an > 1e-10 then
					rm, re, rl = nsum, 0, 0
				else
					rm, re, rl = fromNumberTuple(nsum)
				end
			end
		else
			rm, re, rl = addTuple(m1, e1, l1, m2, e2, l2)
		end
		if rm ~= m1 then bufWriteF64(mBuf, off, rm) end
		if re ~= e1 then bufWriteF64(eBuf, off, re) end
		if rl ~= l1 then bufWriteF64(lBuf, off, rl) end
	end
	return soa
end

function SoA.subScalar(soa: LightNumSoA, scalar: any): LightNumSoA
	local m2, e2, l2 = fastUnpack(scalar)
	-- Subtracting canonical zero is an exact identity.
	if m2 == 0 and e2 == 0 and l2 == 0 then return soa end
	local n = soa.length
	if n == 0 then return soa end
	local mBuf, eBuf, lBuf = soa.m, soa.e, soa.l
	local limit = (n - 1) * 8
	for off = 0, limit, 8 do
		local m1 = bufReadF64(mBuf, off)
		local e1 = bufReadF64(eBuf, off)
		local l1 = bufReadF64(lBuf, off)
		local rm, re, rl
		if l1 == 0 and l2 == 0 then
			-- Exact inline equivalent of subTuple's Layer-0 subtraction branch.
			local ndiff = m1 - m2
			if ndiff == 0 then
				rm, re, rl = 0, 0, 0
			else
				local an = if ndiff > 0 then ndiff else -ndiff
				if an < 1e10 and an > 1e-10 then
					rm, re, rl = ndiff, 0, 0
				else
					rm, re, rl = fromNumberTuple(ndiff)
				end
			end
		else
			rm, re, rl = subTuple(m1, e1, l1, m2, e2, l2)
		end
		if rm ~= m1 then bufWriteF64(mBuf, off, rm) end
		if re ~= e1 then bufWriteF64(eBuf, off, re) end
		if rl ~= l1 then bufWriteF64(lBuf, off, rl) end
	end
	return soa
end

function SoA.mulScalar(soa: LightNumSoA, scalar: any): LightNumSoA
	local m2, e2, l2 = fastUnpack(scalar)
	-- Multiplication by canonical one leaves every valid tuple unchanged.
	if m2 == 1 and e2 == 0 and l2 == 0 then return soa end
	local n = soa.length
	if n == 0 then return soa end
	local mBuf, eBuf, lBuf = soa.m, soa.e, soa.l
	local limit = (n - 1) * 8
	for off = 0, limit, 8 do
		local m1 = bufReadF64(mBuf, off)
		local e1 = bufReadF64(eBuf, off)
		local l1 = bufReadF64(lBuf, off)
		local rm, re, rl
		if l1 == 0 and l2 == 0 then
			-- Exact inline equivalent of mulTuple's Layer-0 * Layer-0 branch.
			local product = m1 * m2
			if product == 0 then
				rm, re, rl = 0, 0, 0
			else
				local ap = if product > 0 then product else -product
				if ap < 1e10 and ap > 1e-10 then
					rm, re, rl = product, 0, 0
				else
					rm, re, rl = fromNumberTuple(product)
				end
			end
		else
			rm, re, rl = mulTuple(m1, e1, l1, m2, e2, l2)
		end
		if rm ~= m1 then bufWriteF64(mBuf, off, rm) end
		if re ~= e1 then bufWriteF64(eBuf, off, re) end
		if rl ~= l1 then bufWriteF64(lBuf, off, rl) end
	end
	return soa
end

function SoA.divScalar(soa: LightNumSoA, scalar: any): LightNumSoA
	local m2, e2, l2 = fastUnpack(scalar)
	-- Division by canonical one leaves every valid tuple unchanged.
	if m2 == 1 and e2 == 0 and l2 == 0 then return soa end
	local n = soa.length
	if n == 0 then return soa end
	local mBuf, eBuf, lBuf = soa.m, soa.e, soa.l
	local limit = (n - 1) * 8
	for off = 0, limit, 8 do
		local m1 = bufReadF64(mBuf, off)
		local e1 = bufReadF64(eBuf, off)
		local l1 = bufReadF64(lBuf, off)
		local rm, re, rl
		if l1 == 0 and l2 == 0 then
			if m2 == 0 then
				if m1 == 0 then rm, re, rl = 0, 1, -1 else rm, re, rl = (if m1 > 0 then 1 else -1), 1, huge end
			elseif m1 == 0 then
				rm, re, rl = 0, 0, 0
			else
				local quotient = m1 / m2
				if quotient == 0 then rm, re, rl = 0, 0, 0
				else
					local aq = if quotient > 0 then quotient else -quotient
					if aq < 1e10 and aq > 1e-10 then rm, re, rl = quotient, 0, 0 else rm, re, rl = fromNumberTuple(quotient) end
				end
			end
		else
			rm, re, rl = divTuple(m1, e1, l1, m2, e2, l2)
		end
		if rm ~= m1 then bufWriteF64(mBuf, off, rm) end
		if re ~= e1 then bufWriteF64(eBuf, off, re) end
		if rl ~= l1 then bufWriteF64(lBuf, off, rl) end
	end
	return soa
end

function SoA.fmaScalar(soa: LightNumSoA, scale: any, offset: any): LightNumSoA
	local sm, se, sl = fastUnpack(scale)
	local om, oe, ol = fastUnpack(offset)
	-- Exact affine identity: x * 1 + 0 leaves every valid tuple unchanged.
	if sm == 1 and se == 0 and sl == 0 and om == 0 and oe == 0 and ol == 0 then
		return soa
	end
	local n = soa.length
	if n == 0 then return soa end
	local mBuf, eBuf, lBuf = soa.m, soa.e, soa.l
	local limit = (n - 1) * 8
	for off = 0, limit, 8 do
		local m1 = bufReadF64(mBuf, off)
		local e1 = bufReadF64(eBuf, off)
		local l1 = bufReadF64(lBuf, off)
		local rm, re, rl
		if l1 == 0 and sl == 0 and ol == 0 then
			local product = m1 * sm
			local ap = if product > 0 then product else -product
			if ap < 1e10 and (ap > 1e-10 or product == 0) then
				local total = product + om
				if total == 0 then rm, re, rl = 0, 0, 0
				else
					local at = if total > 0 then total else -total
					if at < 1e10 and at > 1e-10 then rm, re, rl = total, 0, 0 else rm, re, rl = fromNumberTuple(total) end
				end
			else
				rm, re, rl = fmaTuple(m1, e1, l1, sm, se, sl, om, oe, ol)
			end
		else
			rm, re, rl = fmaTuple(m1, e1, l1, sm, se, sl, om, oe, ol)
		end
		if rm ~= m1 then bufWriteF64(mBuf, off, rm) end
		if re ~= e1 then bufWriteF64(eBuf, off, re) end
		if rl ~= l1 then bufWriteF64(lBuf, off, rl) end
	end
	return soa
end

function SoA.add(dst: LightNumSoA, src: LightNumSoA): LightNumSoA
	local n = mmin(dst.length, src.length)
	if n == 0 then return dst end
	local dmBuf, deBuf, dlBuf = dst.m, dst.e, dst.l
	local smBuf, seBuf, slBuf = src.m, src.e, src.l
	local limit = (n - 1) * 8
	for off = 0, limit, 8 do
		local m1, e1, l1 = bufReadF64(dmBuf, off), bufReadF64(deBuf, off), bufReadF64(dlBuf, off)
		local m2, e2, l2 = bufReadF64(smBuf, off), bufReadF64(seBuf, off), bufReadF64(slBuf, off)
		-- Per-element exact identity avoids entering the arithmetic kernel for sparse zero data.
		if m2 == 0 and e2 == 0 and l2 == 0 then
			continue
		end
		local rm, re, rl
		if l1 == 0 and l2 == 0 then
			local total = m1 + m2
			if total == 0 then rm, re, rl = 0, 0, 0
			else
				local at = if total > 0 then total else -total
				if at < 1e10 and at > 1e-10 then rm, re, rl = total, 0, 0 else rm, re, rl = fromNumberTuple(total) end
			end
		else
			rm, re, rl = addTuple(m1, e1, l1, m2, e2, l2)
		end
		if rm ~= m1 then bufWriteF64(dmBuf, off, rm) end
		if re ~= e1 then bufWriteF64(deBuf, off, re) end
		if rl ~= l1 then bufWriteF64(dlBuf, off, rl) end
	end
	return dst
end

function SoA.mul(dst: LightNumSoA, src: LightNumSoA): LightNumSoA
	local n = mmin(dst.length, src.length)
	if n == 0 then return dst end
	local dmBuf, deBuf, dlBuf = dst.m, dst.e, dst.l
	local smBuf, seBuf, slBuf = src.m, src.e, src.l
	local limit = (n - 1) * 8
	for off = 0, limit, 8 do
		local m1, e1, l1 = bufReadF64(dmBuf, off), bufReadF64(deBuf, off), bufReadF64(dlBuf, off)
		local m2, e2, l2 = bufReadF64(smBuf, off), bufReadF64(seBuf, off), bufReadF64(slBuf, off)
		-- Per-element exact identity avoids entering the arithmetic kernel for sparse one data.
		if m2 == 1 and e2 == 0 and l2 == 0 then
			continue
		end
		local rm, re, rl
		if l1 == 0 and l2 == 0 then
			local product = m1 * m2
			if product == 0 then rm, re, rl = 0, 0, 0
			else
				local ap = if product > 0 then product else -product
				if ap < 1e10 and ap > 1e-10 then rm, re, rl = product, 0, 0 else rm, re, rl = fromNumberTuple(product) end
			end
		else
			rm, re, rl = mulTuple(m1, e1, l1, m2, e2, l2)
		end
		if rm ~= m1 then bufWriteF64(dmBuf, off, rm) end
		if re ~= e1 then bufWriteF64(deBuf, off, re) end
		if rl ~= l1 then bufWriteF64(dlBuf, off, rl) end
	end
	return dst
end

function SoA.fma(dst: LightNumSoA, scaleSoA: LightNumSoA, offsetSoA: LightNumSoA): LightNumSoA
	local n = mmin(dst.length, scaleSoA.length, offsetSoA.length)
	if n == 0 then return dst end
	local dmBuf, deBuf, dlBuf = dst.m, dst.e, dst.l
	local kmBuf, keBuf, klBuf = scaleSoA.m, scaleSoA.e, scaleSoA.l
	local omBuf, oeBuf, olBuf = offsetSoA.m, offsetSoA.e, offsetSoA.l
	local limit = (n - 1) * 8
	for off = 0, limit, 8 do
		local m1, e1, l1 = bufReadF64(dmBuf, off), bufReadF64(deBuf, off), bufReadF64(dlBuf, off)
		local m2, e2, l2 = bufReadF64(kmBuf, off), bufReadF64(keBuf, off), bufReadF64(klBuf, off)
		local m3, e3, l3 = bufReadF64(omBuf, off), bufReadF64(oeBuf, off), bufReadF64(olBuf, off)
		-- Exact per-element affine identity; all three source components are canonical.
		if m2 == 1 and e2 == 0 and l2 == 0 and m3 == 0 and e3 == 0 and l3 == 0 then
			continue
		end
		local rm, re, rl
		if l1 == 0 and l2 == 0 and l3 == 0 then
			local product = m1 * m2
			local ap = if product > 0 then product else -product
			if ap < 1e10 and (ap > 1e-10 or product == 0) then
				local total = product + m3
				if total == 0 then rm, re, rl = 0, 0, 0
				else
					local at = if total > 0 then total else -total
					if at < 1e10 and at > 1e-10 then rm, re, rl = total, 0, 0 else rm, re, rl = fromNumberTuple(total) end
				end
			else
				rm, re, rl = fmaTuple(m1, e1, l1, m2, e2, l2, m3, e3, l3)
			end
		else
			rm, re, rl = fmaTuple(m1, e1, l1, m2, e2, l2, m3, e3, l3)
		end
		if rm ~= m1 then bufWriteF64(dmBuf, off, rm) end
		if re ~= e1 then bufWriteF64(deBuf, off, re) end
		if rl ~= l1 then bufWriteF64(dlBuf, off, rl) end
	end
	return dst
end

function SoA.sumT(soa: LightNumSoA): (number, number, number)
	local n = soa.length
	if n == 0 then return 0, 0, 0 end
	local mBuf, eBuf, lBuf = soa.m, soa.e, soa.l
	local am, ae, al = bufReadF64(mBuf, 0), bufReadF64(eBuf, 0), bufReadF64(lBuf, 0)
	local limit = (n - 1) * 8
	for off = 8, limit, 8 do
		am, ae, al = addTuple(am, ae, al, bufReadF64(mBuf, off), bufReadF64(eBuf, off), bufReadF64(lBuf, off))
	end
	return am, ae, al
end

function SoA.sum(soa: LightNumSoA): buffer
	return pack(SoA.sumT(soa))
end

function SoA.prodT(soa: LightNumSoA): (number, number, number)
	local n = soa.length
	if n == 0 then return 1, 0, 0 end
	local mBuf, eBuf, lBuf = soa.m, soa.e, soa.l
	local am, ae, al = bufReadF64(mBuf, 0), bufReadF64(eBuf, 0), bufReadF64(lBuf, 0)
	local limit = (n - 1) * 8
	for off = 8, limit, 8 do
		am, ae, al = mulTuple(am, ae, al, bufReadF64(mBuf, off), bufReadF64(eBuf, off), bufReadF64(lBuf, off))
	end
	return am, ae, al
end

function SoA.prod(soa: LightNumSoA): buffer
	return pack(SoA.prodT(soa))
end

ln.SoA = SoA

-- ============================================================================
-- ZERO-BUFFER PLAIN-TABLE WRAPPER (v1.30 — MINIMAL WRITE-BARRIER OPERATORS)
-- ============================================================================

local function unpackWrapFast(v: any): (number, number, number)
	local t = type(v)
	if t == "table" then
		local m = v[1]
		if m ~= nil then
			return m, v[2], v[3]
		end
	elseif t == "number" then
		local av = abs(v)
		if v == v and av > 1e-10 and av < 1e10 then
			return v, 0, 0
		end
		return fromNumberTuple(v)
	end
	return fastUnpack(v)
end

-- Inline Layer-0 & Layer-1 metamethod fast paths with zero buffer allocation
function WrapMT.__add(a, b)
	if type(a) == "table" and type(b) == "table" then
		local l1, l2 = a[3], b[3]
		if l1 == 0 and l2 == 0 then
			local n = a[1] + b[1]
			if n == 0 then return setmetatable({ 0, 0, 0 }, WrapMT) end
			local an = if n > 0 then n else -n
			if an < 1e10 and an > 1e-10 then
				return setmetatable({ n, 0, 0 }, WrapMT)
			end
		elseif l1 == 1 and l2 == 1 then
			local m1, e1 = a[1], a[2]
			local m2, e2 = b[1], b[2]
			if e1 < e2 or (e1 == e2 and abs(m1) < abs(m2)) then
				m1, m2 = m2, m1
				e1, e2 = e2, e1
			end
			local diff = e1 - e2
			if diff >= 17 then
				return setmetatable({ m1, e1, 1 }, WrapMT)
			elseif diff == 0 then
				local m = m1 + m2
				local am = if m > 0 then m else -m
				if am >= 1 and am < 10 then
					return setmetatable({ m, e1, 1 }, WrapMT)
				elseif am >= 10 and e1 ~= 9 then
					return setmetatable({ m * 0.1, e1 + 1, 1 }, WrapMT)
				end
			else
				local m = m1 + m2 * (negPow10[diff] or (10 ^ (-diff)))
				local am = if m > 0 then m else -m
				if am >= 1 and am < 10 and e1 ~= -10 then
					return setmetatable({ m, e1, 1 }, WrapMT)
				end
			end
		end
	end
	local m1, e1, l1 = unpackWrapFast(a)
	local m2, e2, l2 = unpackWrapFast(b)
	return rawWrap(addTuple(m1, e1, l1, m2, e2, l2))
end

function WrapMT.__sub(a, b)
	if type(a) == "table" and type(b) == "table" then
		local l1, l2 = a[3], b[3]
		if l1 == 0 and l2 == 0 then
			local n = a[1] - b[1]
			if n == 0 then return setmetatable({ 0, 0, 0 }, WrapMT) end
			local an = if n > 0 then n else -n
			if an < 1e10 and an > 1e-10 then
				return setmetatable({ n, 0, 0 }, WrapMT)
			end
		elseif l1 == 1 and l2 == 1 then
			local m1, e1 = a[1], a[2]
			local m2, e2 = -b[1], b[2]
			if e1 < e2 or (e1 == e2 and abs(m1) < abs(m2)) then
				m1, m2 = m2, m1
				e1, e2 = e2, e1
			end
			local diff = e1 - e2
			if diff >= 17 then
				return setmetatable({ m1, e1, 1 }, WrapMT)
			elseif diff == 0 then
				local m = m1 + m2
				local am = if m > 0 then m else -m
				if am >= 1 and am < 10 then
					return setmetatable({ m, e1, 1 }, WrapMT)
				elseif am >= 10 and e1 ~= 9 then
					return setmetatable({ m * 0.1, e1 + 1, 1 }, WrapMT)
				end
			else
				local m = m1 + m2 * (negPow10[diff] or (10 ^ (-diff)))
				local am = if m > 0 then m else -m
				if am >= 1 and am < 10 and e1 ~= -10 then
					return setmetatable({ m, e1, 1 }, WrapMT)
				end
			end
		end
	end
	local m1, e1, l1 = unpackWrapFast(a)
	local m2, e2, l2 = unpackWrapFast(b)
	return rawWrap(subTuple(m1, e1, l1, m2, e2, l2))
end

function WrapMT.__mul(a, b)
	if type(a) == "table" and type(b) == "table" then
		local l1, l2 = a[3], b[3]
		if l1 == 0 and l2 == 0 then
			local n = a[1] * b[1]
			if n == 0 then return setmetatable({ 0, 0, 0 }, WrapMT) end
			local an = if n > 0 then n else -n
			if an < 1e10 and an > 1e-10 then
				return setmetatable({ n, 0, 0 }, WrapMT)
			end
		elseif l1 == 1 and l2 == 1 then
			local m = a[1] * b[1]
			local e = a[2] + b[2]
			if abs(m) >= 10 then
				m *= 0.1
				e += 1
			end
			if (e >= 10 or e < -10) and abs(e) < huge then
				return setmetatable({ m, e, 1 }, WrapMT)
			end
		end
	end
	local m1, e1, l1 = unpackWrapFast(a)
	local m2, e2, l2 = unpackWrapFast(b)
	return rawWrap(mulTuple(m1, e1, l1, m2, e2, l2))
end

function WrapMT.__div(a, b)
	if type(a) == "table" and type(b) == "table" and a[3] == 1 and b[3] == 1 then
		local m = a[1] / b[1]
		local e = a[2] - b[2]
		if abs(m) < 1 then
			m *= 10
			e -= 1
		end
		if (e >= 10 or e < -10) and abs(e) < huge then
			return setmetatable({ m, e, 1 }, WrapMT)
		end
	end
	local m1, e1, l1 = unpackWrapFast(a)
	local m2, e2, l2 = unpackWrapFast(b)
	return rawWrap(divTuple(m1, e1, l1, m2, e2, l2))
end

function WrapMT.__idiv(a, b)
	local m1, e1, l1 = unpackWrapFast(a)
	local m2, e2, l2 = unpackWrapFast(b)
	return rawWrap(intdivTuple(m1, e1, l1, m2, e2, l2))
end

function WrapMT.__mod(a, b)
	local m1, e1, l1 = unpackWrapFast(a)
	local m2, e2, l2 = unpackWrapFast(b)
	return rawWrap(modTuple(m1, e1, l1, m2, e2, l2))
end

function WrapMT.__pow(a, b)
	local m1, e1, l1 = unpackWrapFast(a)
	local m2, e2, l2 = unpackWrapFast(b)
	return rawWrap(powTuple(m1, e1, l1, m2, e2, l2))
end

function WrapMT.__unm(a)
	if a[3] < 0 then return rawWrap(0, 1, -1) end
	if a[1] == 0 then return rawWrap(0, 0, 0) end
	return rawWrap(-a[1], a[2], a[3])
end

function WrapMT.__eq(a, b)
	local m1, e1, l1 = unpackWrapFast(a)
	local m2, e2, l2 = unpackWrapFast(b)
	return eqTuple(m1, e1, l1, m2, e2, l2)
end

function WrapMT.__lt(a, b)
	if type(a) == "table" and type(b) == "table" and a[3] == 0 and b[3] == 0 then
		return a[1] < b[1]
	end
	local m1, e1, l1 = unpackWrapFast(a)
	local m2, e2, l2 = unpackWrapFast(b)
	return ltTuple(m1, e1, l1, m2, e2, l2)
end

function WrapMT.__le(a, b)
	if type(a) == "table" and type(b) == "table" and a[3] == 0 and b[3] == 0 then
		return a[1] <= b[1]
	end
	local m1, e1, l1 = unpackWrapFast(a)
	local m2, e2, l2 = unpackWrapFast(b)
	return lteTuple(m1, e1, l1, m2, e2, l2)
end

function WrapMT.__tostring(a)
	return ln.tostring(a)
end

-- Fluent immutable arithmetic (zero buffer allocation)
function WrapMT:add(b: any)
	return WrapMT.__add(self, b)
end

function WrapMT:sub(b: any)
	return WrapMT.__sub(self, b)
end

function WrapMT:mul(b: any)
	return WrapMT.__mul(self, b)
end

function WrapMT:div(b: any)
	return WrapMT.__div(self, b)
end

function WrapMT:mod(b: any)
	local m2, e2, l2 = unpackWrapFast(b)
	return rawWrap(modTuple(self[1], self[2], self[3], m2, e2, l2))
end

function WrapMT:pow(b: any)
	local m2, e2, l2 = unpackWrapFast(b)
	return rawWrap(powTuple(self[1], self[2], self[3], m2, e2, l2))
end

-- Minimal write-barrier in-place mutators (writes 0, 1, or 2 slots when e/l stay unchanged)
function WrapMT:addeq(b: any)
	local m2, e2, l2 = unpackWrapFast(b)
	local m1, e1, l1 = self[1], self[2], self[3]
	if l1 == 0 and l2 == 0 then
		local n = m1 + m2
		if n == 0 then
			if m1 ~= 0 then self[1] = 0 end
			return self
		end
		local an = if n > 0 then n else -n
		if an < 1e10 and an > 1e-10 then
			self[1] = n
			return self
		end
	elseif l1 == 1 and l2 == 1 then
		if e1 > e2 or (e1 == e2 and abs(m1) >= abs(m2)) then
			local diff = e1 - e2
			if diff >= 17 then return self end
			if diff == 0 then
				local m = m1 + m2
				local am = if m > 0 then m else -m
				if am >= 1 and am < 10 then
					self[1] = m
					return self
				elseif am >= 10 and e1 ~= 9 then
					self[1] = m * 0.1
					self[2] = e1 + 1
					return self
				end
			else
				local m = m1 + m2 * (negPow10[diff] or (10 ^ (-diff)))
				local am = if m > 0 then m else -m
				if am >= 1 and am < 10 and e1 ~= -10 then
					self[1] = m
					return self
				end
			end
		elseif e2 - e1 >= 17 then
			self[1] = m2
			self[2] = e2
			return self
		end
	end
	local rm, re, rl = addTuple(m1, e1, l1, m2, e2, l2)
	if rm ~= m1 then self[1] = rm end
	if re ~= e1 then self[2] = re end
	if rl ~= l1 then self[3] = rl end
	return self
end

function WrapMT:subeq(b: any)
	local m2, e2, l2 = unpackWrapFast(b)
	local m1, e1, l1 = self[1], self[2], self[3]
	if l1 == 0 and l2 == 0 then
		local n = m1 - m2
		if n == 0 then
			if m1 ~= 0 then self[1] = 0 end
			return self
		end
		local an = if n > 0 then n else -n
		if an < 1e10 and an > 1e-10 then
			self[1] = n
			return self
		end
	elseif l1 == 1 and l2 == 1 then
		local sm2 = -m2
		if e1 > e2 or (e1 == e2 and abs(m1) >= abs(sm2)) then
			local diff = e1 - e2
			if diff >= 17 then return self end
			if diff == 0 then
				local m = m1 + sm2
				local am = if m > 0 then m else -m
				if am >= 1 and am < 10 then
					self[1] = m
					return self
				elseif am >= 10 and e1 ~= 9 then
					self[1] = m * 0.1
					self[2] = e1 + 1
					return self
				end
			else
				local m = m1 + sm2 * (negPow10[diff] or (10 ^ (-diff)))
				local am = if m > 0 then m else -m
				if am >= 1 and am < 10 and e1 ~= -10 then
					self[1] = m
					return self
				end
			end
		elseif e2 - e1 >= 17 then
			self[1] = sm2
			self[2] = e2
			return self
		end
	end
	local rm, re, rl = subTuple(m1, e1, l1, m2, e2, l2)
	if rm ~= m1 then self[1] = rm end
	if re ~= e1 then self[2] = re end
	if rl ~= l1 then self[3] = rl end
	return self
end

function WrapMT:muleq(b: any)
	local m2, e2, l2 = unpackWrapFast(b)
	local m1, e1, l1 = self[1], self[2], self[3]
	if l1 == 0 and l2 == 0 then
		local n = m1 * m2
		if n == 0 then
			if m1 ~= 0 then self[1] = 0 end
			return self
		end
		local an = if n > 0 then n else -n
		if an < 1e10 and an > 1e-10 then
			self[1] = n
			return self
		end
	elseif l1 == 1 and l2 == 0 then
		local m = m1 * m2
		local am = if m > 0 then m else -m
		if am >= 1 and am < 10 and e1 ~= -10 then
			self[1] = m
			return self
		elseif am >= 10 and am < 100 then
			local e = e1 + 1
			if e >= 10 or e < -10 then
				self[1] = m * 0.1
				self[2] = e
				return self
			end
		end
	elseif l1 == 1 and l2 == 1 then
		local m = m1 * m2
		local e = e1 + e2
		if abs(m) >= 10 then
			m *= 0.1
			e += 1
		end
		if (e >= 10 or e < -10) and abs(e) < huge then
			self[1] = m
			self[2] = e
			return self
		end
	end
	local rm, re, rl = mulTuple(m1, e1, l1, m2, e2, l2)
	if rm ~= m1 then self[1] = rm end
	if re ~= e1 then self[2] = re end
	if rl ~= l1 then self[3] = rl end
	return self
end

function WrapMT:diveq(b: any)
	local m2, e2, l2 = unpackWrapFast(b)
	local m1, e1, l1 = self[1], self[2], self[3]
	if l1 == 0 and l2 == 0 and m2 ~= 0 then
		local n = m1 / m2
		if n == 0 then
			if m1 ~= 0 then self[1] = 0 end
			return self
		end
		local an = if n > 0 then n else -n
		if an < 1e10 and an > 1e-10 then
			self[1] = n
			return self
		end
	elseif l1 == 1 and l2 == 1 then
		local m = m1 / m2
		local e = e1 - e2
		if abs(m) < 1 then
			m *= 10
			e -= 1
		end
		if (e >= 10 or e < -10) and abs(e) < huge then
			self[1] = m
			self[2] = e
			return self
		end
	end
	local rm, re, rl = divTuple(m1, e1, l1, m2, e2, l2)
	if rm ~= m1 then self[1] = rm end
	if re ~= e1 then self[2] = re end
	if rl ~= l1 then self[3] = rl end
	return self
end

function WrapMT:fmaeq(a: any, b: any)
	local ma, ea, la = unpackWrapFast(a)
	local mb, eb, lb = unpackWrapFast(b)
	local m1, e1, l1 = self[1], self[2], self[3]
	if l1 == 0 and la == 0 and lb == 0 then
		local p = ma * mb
		local ap = if p > 0 then p else -p
		if ap < 1e10 and (ap > 1e-10 or p == 0) then
			local n = m1 + p
			if n == 0 then
				if m1 ~= 0 then self[1] = 0 end
				return self
			end
			local an = if n > 0 then n else -n
			if an < 1e10 and an > 1e-10 then
				self[1] = n
				return self
			end
		end
	end
	local rm, re, rl = fmaTuple(ma, ea, la, mb, eb, lb, m1, e1, l1)
	if rm ~= m1 then self[1] = rm end
	if re ~= e1 then self[2] = re end
	if rl ~= l1 then self[3] = rl end
	return self
end

function WrapMT:poweq(b: any)
	local m2, e2, l2 = unpackWrapFast(b)
	local m1, e1, l1 = self[1], self[2], self[3]
	local rm, re, rl = powTuple(m1, e1, l1, m2, e2, l2)
	if rm ~= m1 then self[1] = rm end
	if re ~= e1 then self[2] = re end
	if rl ~= l1 then self[3] = rl end
	return self
end

function WrapMT:tetreq(b: any)
	return ln.tetreq(self, b)
end

function WrapMT:modeq(b: any)
	local m2, e2, l2 = unpackWrapFast(b)
	self[1], self[2], self[3] = modTuple(self[1], self[2], self[3], m2, e2, l2)
	return self
end

function WrapMT:flooreq()
	self[1], self[2], self[3] = floorTuple(self[1], self[2], self[3])
	return self
end

function WrapMT:ceileq()
	self[1], self[2], self[3] = ceilTuple(self[1], self[2], self[3])
	return self
end

function WrapMT:roundeq()
	self[1], self[2], self[3] = roundTuple(self[1], self[2], self[3])
	return self
end

function WrapMT:negeq()
	if self[3] >= 0 and self[1] ~= 0 then self[1] = -self[1] end
	return self
end

function WrapMT:abseq()
	if self[1] < 0 then self[1] = -self[1] end
	return self
end

function WrapMT:eq(b: any): boolean
	local m2, e2, l2 = unpackWrapFast(b)
	return eqTuple(self[1], self[2], self[3], m2, e2, l2)
end

function WrapMT:lt(b: any): boolean
	local m2, e2, l2 = unpackWrapFast(b)
	return ltTuple(self[1], self[2], self[3], m2, e2, l2)
end

function WrapMT:lte(b: any): boolean
	local m2, e2, l2 = unpackWrapFast(b)
	return lteTuple(self[1], self[2], self[3], m2, e2, l2)
end

function WrapMT:gt(b: any): boolean
	local m2, e2, l2 = unpackWrapFast(b)
	return gtTuple(self[1], self[2], self[3], m2, e2, l2)
end

function WrapMT:gte(b: any): boolean
	local m2, e2, l2 = unpackWrapFast(b)
	return gteTuple(self[1], self[2], self[3], m2, e2, l2)
end

function WrapMT:compare(b: any): number?
	local m2, e2, l2 = unpackWrapFast(b)
	return compareTuples(self[1], self[2], self[3], m2, e2, l2)
end

function WrapMT:clone()
	return rawWrap(self[1], self[2], self[3])
end

function WrapMT:toTable(): { number }
	return { self[1], self[2], self[3] }
end

function WrapMT:to25Buffer(): buffer
	return pack(self[1], self[2], self[3])
end

function WrapMT:unwrap(): buffer
	return pack17(self[1], self[2], self[3])
end

function WrapMT:toNumber(): number return tupleToNumber(self[1], self[2], self[3]) end
function WrapMT:toSuffix(): string return sufPrecise(self[1], self[2], self[3]) end
function WrapMT:toScientific(): string return sciPrecise(self[1], self[2], self[3]) end
function WrapMT:toEngineer(): string return engPrecise(self[1], self[2], self[3]) end
function WrapMT:toLayered(): string return ln.toLayered(self) end
function WrapMT:toHyper(): string return ln.toHyper(self) end

function ln.wrap(n: any)
	return rawWrap(fastUnpack(n))
end
ln.fast = ln.wrap
ln.obj = ln.wrap

-- ============================================================================
-- CONSTANTS
-- ============================================================================

ln.Constants = {
	Zero = ln.new(0),
	One = ln.new(1, 0, 1),
	Two = ln.new(1, 0, 2),
	Ten = ln.new(1, 0, 10),
	E = ln.new(1, 0, EULER),
	Pi = ln.new(1, 0, pi),
	Googol = ln.new(1, 1, 100),
	Googolplex = ln.new(1, 2, 100),
	Inf = ln.new(1, huge, 1),
	NegInf = ln.new(-1, huge, 1),
	NaN = ln.new(1, -1, 1),
	MaxNumber = ln.new(1, 1, INF_LIMIT),
}



-- ============================================================================
-- v1.8 EXTENDED LAYER ENGINE (opt-in; range and behavior preserved in v1.9.1)
-- ============================================================================
-- Hyper values use a decimal-string layer index, capped at 400 digits in v1.9.1.
-- The finite mantissa/exponent remain IEEE-754 numbers. This extends the
-- *layer counter*, not the precision of the mantissa or exponent.
-- Values at layers >= 3 follow LightNum's existing dominant-term arithmetic.
local Hyper = {}
local HyperMT = {}
HyperMT.__index = HyperMT
local HYPER_TAG = "LightNum.Hyper.v1"
local MAX_HYPER_LAYER_DIGITS = 400

local function hDigits(v: any): string
	if v == "-1" or v == "INF" then return v end
	local t = type(v)
	if t == "number" then
		if v ~= v or v < 0 or v == huge or v ~= floor(v) then error("Hyper layer must be a non-negative integer") end
		if v < 9007199254740991 then return string.format("%.0f", v) end
		error("Layer number exceeds exact integer input; pass a decimal string")
	elseif t ~= "string" then
		error("Hyper layer must be a decimal string or exact number")
	end
	if #v == 0 or #v > MAX_HYPER_LAYER_DIGITS or string.match(v, "^%d+$") == nil then error("Invalid Hyper layer string (maximum 400 decimal digits)") end
	local z = string.gsub(v, "^0+", "")
	return if z == "" then "0" else z
end

local function hCmpInt(a: string, b: string): number
	if #a < #b then return -1 elseif #a > #b then return 1 end
	if a < b then return -1 elseif a > b then return 1 end
	return 0
end

local function hAddInt(a: string, b: string): string
	local i, j, carry = #a, #b, 0
	local maxLen = math.max(i, j) + 1
	local out = tcreate(maxLen)
	local pos = maxLen
	while i > 0 or j > 0 or carry > 0 do
		local x = if i > 0 then string.byte(a, i) - 48 else 0
		local y = if j > 0 then string.byte(b, j) - 48 else 0
		local n = x + y + carry
		out[pos] = string.char(48 + n % 10)
		carry = math.floor(n / 10); i -= 1; j -= 1; pos -= 1
	end
	return tconcat(out, "", pos + 1, maxLen)
end

local function hSubInt(a: string, b: string): string -- requires a >= b
	local i, j, borrow = #a, #b, 0
	local out = tcreate(i)
	while i > 0 do
		local x = string.byte(a, i) - 48 - borrow
		local y = if j > 0 then string.byte(b, j) - 48 else 0
		borrow = 0
		if x < y then x += 10; borrow = 1 end
		out[i] = string.char(48 + x - y)
		i -= 1; j -= 1
	end
	local r = string.gsub(tconcat(out), "^0+", "")
	return if r == "" then "0" else r
end

local function hNew(m: number, e: number, layer: any)
	if type(layer) == "number" then
		if layer < 0 then layer = "-1" elseif layer == huge then layer = "INF" end
	end
	local l = hDigits(layer)
	if m ~= m or e ~= e then m, e, l = 0, 1, "-1" end
	if l == "-1" then return setmetatable({_tag=HYPER_TAG,m=0,e=1,l="-1"}, HyperMT) end
	if m == 0 then return setmetatable({_tag=HYPER_TAG,m=0,e=0,l="0"}, HyperMT) end
	if m == huge or e == huge then return setmetatable({_tag=HYPER_TAG,m=(if m < 0 then -1 else 1),e=1,l="INF"}, HyperMT) end
	if e == -huge then return setmetatable({_tag=HYPER_TAG,m=0,e=0,l="0"}, HyperMT) end
	-- Canonical sign-only mantissa at the very high layers, matching v1.7.
	if hCmpInt(l, "3") >= 0 then m = if m < 0 then -1 else 1 end
	return setmetatable({_tag=HYPER_TAG,m=m,e=e,l=l}, HyperMT)
end

local function hIs(x: any): boolean
	return type(x) == "table" and rawget(x, "_tag") == HYPER_TAG
end

local function hFrom(x: any)
	if hIs(x) then return x end
	local m,e,l = fastUnpack(x)
	local layer = if l < 0 then "-1" elseif l == huge then "INF" else hDigits(l)
	return hNew(m,e,layer)
end

local function hMagCmp(a: any, b: any): number
	if a.l == "INF" or b.l == "INF" then
		if a.l == b.l then return 0 end
		return if a.l == "INF" then 1 else -1
	end
	local function rank(v: any): string
		if v.l == "0" then return if math.abs(v.m) >= 1 then "0" else "-1" end
		if v.l == "1" then return if v.e >= 0 then "1" else "-1" end
		return if v.e > 0 then v.l else "-"..v.l
	end
	local ar,br=rank(a),rank(b)
	local c: number
	local aneg=string.sub(ar,1,1)=="-"
	local bneg=string.sub(br,1,1)=="-"
	if aneg and not bneg then c=-1
	elseif bneg and not aneg then c=1
	elseif aneg then c=-hCmpInt(string.sub(ar,2),string.sub(br,2))
	else c=hCmpInt(ar,br) end
	if c ~= 0 then return c end
	-- Match v1.7's exact ordering for the native layer 0/1 region.
	if (a.l == "0" or a.l == "1") and (b.l == "0" or b.l == "1") then
		return compareMagnitude(a.m,a.e,tonumber(a.l),b.m,b.e,tonumber(b.l))
	end
	if a.e < b.e then return -1 elseif a.e > b.e then return 1 end
	local am,bm=math.abs(a.m),math.abs(b.m)
	return if am < bm then -1 elseif am > bm then 1 else 0
end
function Hyper.is(x: any): boolean return hIs(x) end
function Hyper.from(x: any) return hFrom(x) end
function Hyper.fromParts(m: number, e: number, layer: any) return hNew(m,e,layer) end
function Hyper.layer(x: any): string return hFrom(x).l end
function Hyper.totuple(x: any): (number, number, string)
	local v=hFrom(x); return v.m,v.e,v.l
end
function Hyper.addLayerIndex(a: any, b: any): string
	local aa,bb=hDigits(a),hDigits(b)
	if aa=="INF" or bb=="INF" or aa=="-1" or bb=="-1" then error("Layer index arithmetic requires non-negative finite integers") end
	return hDigits(hAddInt(aa,bb))
end
function Hyper.subtractLayerIndex(a: any, b: any): string
	local aa,bb=hDigits(a),hDigits(b)
	if aa=="INF" or bb=="INF" or aa=="-1" or bb=="-1" then error("Layer index arithmetic requires non-negative finite integers") end
	if hCmpInt(aa,bb)<0 then error("Layer index subtraction would be negative") end
	return hSubInt(aa,bb)
end
function Hyper.raiseLayer(x: any, delta: any)
	local v=hFrom(x)
	if v.l=="INF" or v.l=="-1" then error("Cannot raise an infinite or NaN layer") end
	local d=hDigits(delta)
	if d=="INF" or d=="-1" then error("Layer delta must be a finite non-negative integer") end
	return hNew(v.m,v.e,hAddInt(v.l,d))
end
function Hyper.sign(x: any): number local v=hFrom(x); return if v.m > 0 then 1 elseif v.m < 0 then -1 else 0 end
function Hyper.compare(x: any, y: any): number?
	local a,b=hFrom(x),hFrom(y)
	if a.l == "-1" or b.l == "-1" then return nil end
	if a.m == 0 then return if b.m == 0 then 0 else (if b.m > 0 then -1 else 1) end
	if b.m == 0 then return if a.m > 0 then 1 else -1 end
	if (a.m > 0) ~= (b.m > 0) then return if a.m > 0 then 1 else -1 end
	local c=hMagCmp(a,b)
	return if a.m > 0 then c else -c
end
function Hyper.eq(x: any,y: any): boolean return Hyper.compare(x,y)==0 end
function Hyper.lt(x: any,y: any): boolean local c=Hyper.compare(x,y); return c ~= nil and c < 0 end
function Hyper.lte(x: any,y: any): boolean local c=Hyper.compare(x,y); return c ~= nil and c <= 0 end
function Hyper.gt(x: any,y: any): boolean local c=Hyper.compare(x,y); return c ~= nil and c > 0 end
function Hyper.gte(x: any,y: any): boolean local c=Hyper.compare(x,y); return c ~= nil and c >= 0 end

local function hDominant(a: any,b: any, sign: number)
	local c=hMagCmp(a,b)
	local v=if c >= 0 then a else b
	return hNew(sign,v.e,v.l)
end
function Hyper.add(x: any,y: any)
	local a,b=hFrom(x),hFrom(y)
	if a.l == "-1" or b.l == "-1" then return hNew(0,1,"-1") end
	if a.m == 0 then return hNew(b.m,b.e,b.l) end
	if b.m == 0 then return hNew(a.m,a.e,a.l) end
	local c=hMagCmp(a,b)
	if c == 0 and ((a.m > 0) ~= (b.m > 0)) then return hNew(0,0,"0") end
	-- Preserve exact v1.7 arithmetic for operands that fit its native layer index.
	if #a.l < 16 and #b.l < 16 then
		local la,lb=tonumber(a.l),tonumber(b.l)
		if la and lb and la <= 1e15 and lb <= 1e15 then
			local m,e,l=addTuple(a.m,a.e,la,b.m,b.e,lb)
			return hNew(m,e,l)
		end
	end
	local dominant = if c >= 0 then a else b
	return hDominant(a,b,if dominant.m > 0 then 1 else -1)
end
function Hyper.sub(x: any,y: any)
	local b=hFrom(y); return Hyper.add(x,hNew(-b.m,b.e,b.l))
end
function Hyper.mul(x: any,y: any)
	local a,b=hFrom(x),hFrom(y)
	if a.l == "-1" or b.l == "-1" then return hNew(0,1,"-1") end
	if a.m == 0 or b.m == 0 then return hNew(0,0,"0") end
	local sign=if (a.m > 0)==(b.m > 0) then 1 else -1
	if a.l == "INF" or b.l == "INF" then return hNew(sign,1,"INF") end
	if hCmpInt(a.l,"3") >= 0 or hCmpInt(b.l,"3") >= 0 then
		local c=hCmpInt(a.l,b.l)
		local v=if c > 0 then a elseif c < 0 then b elseif math.abs(a.e) >= math.abs(b.e) then a else b
		return hNew(sign,v.e,v.l)
	end
	local la,lb=tonumber(a.l),tonumber(b.l)
	local m,e,l=mulTuple(a.m,a.e,la,b.m,b.e,lb)
	return hNew(m,e,l)
end
function Hyper.div(x: any,y: any)
	local a,b=hFrom(x),hFrom(y)
	if a.l == "-1" or b.l == "-1" or (a.m==0 and b.m==0) then return hNew(0,1,"-1") end
	if b.m == 0 then return hNew(if a.m>0 then 1 else -1,1,"INF") end
	if a.m == 0 then return hNew(0,0,"0") end
	local sign=if (a.m > 0)==(b.m > 0) then 1 else -1
	if a.l == "INF" then return hNew(sign,1,"INF") end
	if b.l == "INF" then return hNew(0,0,"0") end
	if hCmpInt(a.l,"3") >= 0 or hCmpInt(b.l,"3") >= 0 then
		local c=hCmpInt(a.l,b.l)
		if c > 0 then return hNew(sign,a.e,a.l) end
		if c < 0 then return hNew(sign,-b.e,b.l) end
		if math.abs(a.e) >= math.abs(b.e) then return hNew(sign,a.e,a.l) end
		return hNew(sign,-b.e,b.l)
	end
	local m,e,l=divTuple(a.m,a.e,tonumber(a.l),b.m,b.e,tonumber(b.l))
	return hNew(m,e,l)
end
function Hyper.pow(x: any,y: any)
	local a,b=hFrom(x),hFrom(y)
	if a.l=="-1" or b.l=="-1" then return hNew(0,1,"-1") end
	if b.m==0 then return hNew(1,0,"0") end
	if a.m==0 then return hNew(0,0,"0") end
	if hCmpInt(a.l,"3") >= 0 then
		if b.l=="0" and b.e==0 and b.m==1 then return hNew(a.m,a.e,a.l) end
		if b.l=="0" and b.e==0 and b.m==-1 then return hNew(1,-a.e,a.l) end
		error("Hyper.pow for layers >= 3 currently supports exponents +1 and -1 only; refusing an unverified approximation")
	end
	if b.l=="0" and b.e==0 and math.abs(b.m)<1e6 then
		local m,e,l=powTuple(a.m,a.e,tonumber(a.l),b.m,0,0)
		return hNew(m,e,l)
	end
	return hNew(1,1,"INF")
end
function Hyper.toLightNum(x: any): buffer
	local a=hFrom(x)
	if a.l=="INF" or a.l=="-1" or #a.l > 15 then error("Hyper value cannot be represented by a standard LightNum buffer") end
	local l=tonumber(a.l)
	if not l or l > 1e15 then error("Hyper layer exceeds standard LightNum range") end
	return pack(a.m,a.e,l)
end
function Hyper.toTable(x: any)
	local a=hFrom(x); return { _tag=HYPER_TAG, m=a.m, e=a.e, l=a.l }
end
function Hyper.serialize(x: any): string
	local a=hFrom(x)
	return "LNH1|"..tostring(a.m).."|"..tostring(a.e).."|"..a.l
end
function Hyper.deserialize(s: string)
	if type(s)~="string" then error("Hyper.deserialize expects a string") end
	local m,e,l=string.match(s,"^LNH1|([^|]+)|([^|]+)|([^|]+)$")
	if not m then error("Invalid Hyper serialization") end
	local mn,en=tonumber(m),tonumber(e)
	if mn == nil or en == nil then error("Invalid Hyper numeric fields") end
	return hNew(mn,en,l)
end
function Hyper.toString(x: any): string
	local a=hFrom(x)
	if a.l=="-1" then return "NaN" end
	if a.l=="INF" then return if a.m<0 then "-Infinity" else "Infinity" end
	if a.m==0 then return "0" end
	if a.l=="0" or a.l=="1" or a.l=="2" then
		return ln.showPreciseS(pack(a.m,a.e,tonumber(a.l)))
	end
	return (if a.m<0 then "-" else "").."10^^"..a.l.."("..string.format("%.17g",a.e)..")"
end
function Hyper.__index(self: any,key: string): any
	if key=="toString" then return function(v) return Hyper.toString(v) end end
	return Hyper[key]
end
HyperMT.__index=Hyper.__index
HyperMT.__tostring=function(v) return Hyper.toString(v) end
HyperMT.__add=function(a,b) return Hyper.add(a,b) end
HyperMT.__sub=function(a,b) return Hyper.sub(a,b) end
HyperMT.__mul=function(a,b) return Hyper.mul(a,b) end
HyperMT.__div=function(a,b) return Hyper.div(a,b) end
HyperMT.__pow=function(a,b) return Hyper.pow(a,b) end
HyperMT.__unm=function(a) local v=hFrom(a); return hNew(-v.m,v.e,v.l) end
HyperMT.__eq=Hyper.eq
HyperMT.__lt=Hyper.lt
HyperMT.__le=Hyper.lte
ln.Hyper = Hyper
ln.hyper = Hyper.from


return ln
