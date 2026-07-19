# Phase A: TypeScript Foundations (In-Repo Lessons)

**Maps to:** [Blueprint Phase A](../full_stack_ai_engineer_roadmap.md#phase-a-typescript-foundations)

**Goal:** Write small programs with correct types so later APIs and React/Next code stay maintainable.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Lesson 1: Tooling and your first types](#lesson-1-tooling-and-your-first-types)
- [Lesson 2: Functions and narrowing](#lesson-2-functions-and-narrowing)
- [Lesson 3: Objects, arrays, and immutability patterns](#lesson-3-objects-arrays-and-immutability-patterns)
- [Lesson 4: Unions, intersections, and literals](#lesson-4-unions-intersections-and-literals)
- [Lesson 5: Generics (the practical slice)](#lesson-5-generics-the-practical-slice)
- [Lesson 6: Classes, interfaces, and guards](#lesson-6-classes-interfaces-and-guards)
- [Self-check exercises](#self-check-exercises)
- [Next step](#next-step)

---

## Prerequisites

- Comfortable running commands in a terminal.
- Basic programming (variables, loops). If you only know Python. That is enough to start.

Install **Node.js** (using [nvm](https://github.com/nvm-sh/nvm#installing-and-updating) is recommended so you can switch versions per project). Then:

```bash
mkdir ts-phase-a && cd ts-phase-a
npm init -y
npm install -D typescript ts-node @types/node
npx tsc --init --rootDir src --outDir dist --esModuleInterop --resolveJsonModule --strict
mkdir -p src
```

Create `src/lesson01.ts` and run with `npx ts-node src/lesson01.ts` (or compile with `npx tsc` and run `node dist/lesson01.js`).

---

## Lesson 1: Tooling and your first types

TypeScript adds a **type layer** that disappears at runtime. The compiler catches mistakes early.

**Core primitives:** `string`, `number`, `boolean`, `null`, `undefined`, `bigint`, `symbol`.

```ts
const userName: string = "Ada";
let score: number = 0;
score = 10;
```

**Type inference:** when the compiler can prove a type, you may omit annotations.

```ts
const inferred = 42; // number
```

**Arrays and tuples:**

```ts
const tags: string[] = ["ml", "typescript"];
const pair: [string, number] = ["epoch", 1];
```

**Optional properties:**

```ts
type User = { id: string; displayName?: string };
const u: User = { id: "u1" };
```

**Exercise:** Define a `Book` type with `title`, `year`, and optional `isbn`. Create two values: one with `isbn`, one without.

---

## Lesson 2: Functions and narrowing

**Parameter and return types:**

```ts
function add(a: number, b: number): number {
  return a + b;
}
```

**Union types** mean “one of several shapes.” You often **narrow** with checks:

```ts
type Result = { ok: true; value: number } | { ok: false; error: string };

function unwrap(r: Result): number {
  if (r.ok) return r.value;
  throw new Error(r.error);
}
```

**Optional chaining and nullish coalescing:**

```ts
const len = u.displayName?.length ?? 0;
```

**Exercise:** Write a function `formatScore` that accepts `number | null` and returns a string. If `null`, return `"—"`.

---

## Lesson 3: Objects, arrays, and immutability patterns

**Readonly** helps express intent:

```ts
type Config = Readonly<{ apiBaseUrl: string; timeoutMs: number }>;
```

**Spread and rest:**

```ts
const defaults = { retries: 3, timeoutMs: 5000 };
const runtime = { ...defaults, timeoutMs: 8000 };
const { retries, ...rest } = runtime;
```

**Destructuring with types:**

```ts
function describe({ title, year }: { title: string; year: number }): string {
  return `${title} (${year})`;
}
```

**Exercise:** Given `const items = [{ id: 1 }, { id: 2 }]`, write a function that returns a **new** array with one more item `{ id: 3 }` without mutating `items`.

---

## Lesson 4: Unions, intersections, and literals

**Literal types** pin exact values:

```ts
type Theme = "light" | "dark";
```

**Intersection** combines shapes (use sparingly; prefer composition when it gets hard to read):

```ts
type Timestamps = { createdAt: string; updatedAt: string };
type Entity = { id: string } & Timestamps;
```

**`as const`** freezes literal types:

```ts
const routes = ["/health", "/users"] as const;
type Route = (typeof routes)[number];
```

**Exercise:** Model an API error as `{ code: "NOT_FOUND" } | { code: "RATE_LIMIT"; retryAfterSec: number }`. Write a `logError` function that prints different messages for each case.

---

## Lesson 5: Generics (the practical slice)

Generics let you reuse logic while keeping types precise.

```ts
function first<T>(items: T[]): T | undefined {
  return items[0];
}
```

**Constraint:**

```ts
function getId<T extends { id: string }>(obj: T): string {
  return obj.id;
}
```

**Exercise:** Implement `groupBy` that takes an array of `{ id: string; tag: string }` and returns `Record<string, string[]>` mapping each `tag` to a list of `id`s.

---

## Lesson 6: Classes, interfaces, and guards

**Interfaces** describe object shapes; **classes** are runtime values with types.

```ts
interface Logger {
  info(message: string): void;
}

class ConsoleLogger implements Logger {
  info(message: string): void {
    console.log(message);
  }
}
```

**Type guards:**

```ts
function isString(x: unknown): x is string {
  return typeof x === "string";
}
```

**Exercise:** Create an `HttpError` class with `statusCode` and `message`. Write a function `toUserMessage(err: unknown): string` that returns a friendly string for `HttpError` and a generic fallback otherwise.

---

## Self-check exercises

1. Implement a tiny `Result<T, E>` type and helpers `ok` / `err` without using `any`.
2. Parse JSON safely: write `parseJsonObject(input: string): Record<string, unknown> | null` that returns `null` on invalid JSON or non-object roots.
3. Model pagination: `{ page: number; pageSize: number }` and a function `offset` that returns SQL-style offset.

---

## Next step

Continue to [Phase B: Node & APIs](phase-b-node-apis.md) after you can comfortably pass the exercises above without guessing types.
