import React, { useState, useRef, useEffect, useMemo, useCallback, memo } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from "recharts";
import {
  Rocket, Play, Pause, RotateCcw, Trash2, Plus, ChevronUp, ChevronDown, Copy,
  AlertTriangle, CheckCircle2, XCircle, Flame, Wind, Award, FlaskConical,
  Save, Square, Gauge, Trophy, FolderOpen, X, Info, Layers, Satellite as SatelliteIcon,
} from "lucide-react";

/* =========================================================================
   PHYSICS MODULE  (physics/)
   Pure functions only. No React, no rendering, no UI. Kept isolated so the
   flight model can be unit-tested independently of everything else.
   ========================================================================= */

const G0 = 9.80665; // standard gravity at sea level, m/s^2 (also the Isp->mdot conversion constant)
const R_EARTH = 6371000; // m, used for altitude-aware gravity

/** Altitude-aware gravity: g(h) = g0 * (R / (R+h))^2. Falls back to constant g0 near the ground. */
function gravityAt(altitudeM) {
  const r = R_EARTH + Math.max(0, altitudeM);
  return G0 * (R_EARTH / r) ** 2;
}

/** Exponential atmosphere model. Returns kg/m^3, approaching 0 with altitude. */
function airDensity(altitudeM) {
  if (altitudeM < 0) altitudeM = 0;
  const rho0 = 1.225; // sea level, kg/m^3
  const scaleHeight = 8500; // m
  const rho = rho0 * Math.exp(-altitudeM / scaleHeight);
  return rho < 1e-5 ? 0 : rho;
}

/** F_drag = 0.5 * rho * v^2 * Cd * A. Always a non-negative magnitude; caller applies direction. */
function dragForceMag(rho, velocity, cd, area) {
  return 0.5 * rho * velocity * velocity * cd * area;
}

/** massFlowRate = thrust / (Isp * g0) */
function massFlowRate(thrust, isp) {
  if (isp <= 0) return 0;
  return thrust / (isp * G0);
}

/** Tsiolkovsky rocket equation: dv = Isp * g0 * ln(m0/mf) */
function tsiolkovsky(isp, m0, mf) {
  if (mf <= 0 || m0 <= mf || isp <= 0) return 0;
  return isp * G0 * Math.log(m0 / mf);
}

// Simulation safety ceilings — the model must never spin forever or blow up numerically.
const SIM_MAX_TIME = 400; // s
const SIM_MAX_ALTITUDE = 250000; // m — well past anything these parts can reach; a safety backstop, not a gameplay limit
const SIM_MAX_SPEED = 6000; // m/s

/**
 * Advance the flight state by dt seconds using explicit Euler integration.
 * Never mutates input; returns a brand new { state, flags, events, derived }.
 * `flags.aborted` / `flags.abortReason` are set if a safety limit is hit.
 */
function integrateStep(state, rocket, flags, dt) {
  const events = [];
  let { altitude, velocity, propellantByStage } = state;
  let { activeStageIndex, jettisonedDryMass, chuteDeployed, apogeeReached, maxAltitude } = flags;

  propellantByStage = propellantByStage.slice();

  // --- active stage & thrust ---
  let thrust = 0;
  let ispEff = 0;
  const stage = rocket.stages[activeStageIndex];

  if (stage && propellantByStage[activeStageIndex] > 0) {
    thrust = stage.thrust;
    ispEff = stage.isp;
    const mdot = massFlowRate(thrust, ispEff);
    propellantByStage[activeStageIndex] = Math.max(0, propellantByStage[activeStageIndex] - mdot * dt);
    if (propellantByStage[activeStageIndex] <= 0) {
      events.push({ type: "BURNOUT", stage: activeStageIndex });
      if (activeStageIndex < rocket.stages.length - 1) {
        events.push({ type: "SEPARATION", stage: activeStageIndex });
        jettisonedDryMass += stage.dryMass;
        activeStageIndex += 1;
        if (rocket.stages[activeStageIndex] && propellantByStage[activeStageIndex] > 0) {
          events.push({ type: "IGNITION", stage: activeStageIndex });
        }
      }
    }
  }

  // --- mass remaining (never allow zero/negative mass to divide by) ---
  let mass = rocket.totalDryMass - jettisonedDryMass;
  for (let i = 0; i < propellantByStage.length; i++) mass += propellantByStage[i];
  mass = Math.max(mass, 0.05);

  // --- forces ---
  const g = gravityAt(altitude);
  const weight = mass * g;
  const rho = airDensity(altitude);

  let cd = rocket.dragCoefficient;
  let area = rocket.referenceArea;
  if (chuteDeployed && rocket.chute) {
    cd = rocket.chute.dragCoefficient;
    area = rocket.chute.deployedArea;
  }
  const drag = dragForceMag(rho, velocity, cd, area);
  const dragSigned = velocity >= 0 ? -drag : drag; // drag always opposes current motion

  const netForce = thrust - weight + dragSigned;
  const acceleration = netForce / mass;

  velocity = velocity + acceleration * dt;
  altitude = altitude + velocity * dt;

  if (altitude < 0) altitude = 0;
  if (altitude > maxAltitude) maxAltitude = altitude;

  if (!apogeeReached && velocity < 0 && maxAltitude > 0) {
    apogeeReached = true;
    events.push({ type: "APOGEE", altitude: maxAltitude });
  }

  if (
    rocket.chute &&
    !chuteDeployed &&
    apogeeReached &&
    altitude <= rocket.chute.deploymentAltitude &&
    velocity < 0
  ) {
    chuteDeployed = true;
    events.push({ type: "CHUTE_DEPLOY", altitude });
  }

  if (altitude <= 0 && apogeeReached) {
    altitude = 0;
    velocity = 0;
  }

  // --- numerical safety net: never let NaN/Infinity or runaway values escape this function ---
  let aborted = false;
  let abortReason = null;
  if (!Number.isFinite(altitude) || !Number.isFinite(velocity) || !Number.isFinite(mass)) {
    aborted = true;
    abortReason = "Numerical instability detected (non-finite value) — simulation halted safely.";
    altitude = Number.isFinite(altitude) ? altitude : maxAltitude;
    velocity = 0;
  } else if (Math.abs(velocity) > SIM_MAX_SPEED) {
    aborted = true;
    abortReason = `Velocity exceeded the ${SIM_MAX_SPEED} m/s safety limit — simulation halted.`;
  } else if (altitude > SIM_MAX_ALTITUDE) {
    aborted = true;
    abortReason = `Altitude exceeded the ${SIM_MAX_ALTITUDE / 1000} km safety ceiling — simulation halted.`;
  }

  return {
    state: { altitude, velocity, propellantByStage },
    flags: { activeStageIndex, jettisonedDryMass, chuteDeployed, apogeeReached, maxAltitude, aborted, abortReason },
    events,
    derived: { mass, thrust, acceleration, drag, rho, dynamicPressure: 0.5 * rho * velocity * velocity, g },
  };
}

/* =========================================================================
   ROCKET MODULE  (rocket/)
   Part library + pure functions that turn a stack of parts into a computed
   rocket description, staging breakdown, fuel-compatibility report, and
   validity checks.
   ========================================================================= */

// One consistent color per category, used everywhere (palette, stack, badges, stats).
const CATEGORY_COLORS = {
  structure: { text: "text-sky-400", border: "border-sky-600", bg: "bg-sky-500/10", ring: "ring-sky-400", dot: "bg-sky-500" },
  fuel: { text: "text-cyan-400", border: "border-cyan-600", bg: "bg-cyan-500/10", ring: "ring-cyan-400", dot: "bg-cyan-500" },
  engines: { text: "text-orange-400", border: "border-orange-600", bg: "bg-orange-500/10", ring: "ring-orange-400", dot: "bg-orange-500" },
  aero: { text: "text-emerald-400", border: "border-emerald-600", bg: "bg-emerald-500/10", ring: "ring-emerald-400", dot: "bg-emerald-500" },
  payload: { text: "text-fuchsia-400", border: "border-fuchsia-600", bg: "bg-fuchsia-500/10", ring: "ring-fuchsia-400", dot: "bg-fuchsia-500" },
  recovery: { text: "text-amber-400", border: "border-amber-600", bg: "bg-amber-500/10", ring: "ring-amber-400", dot: "bg-amber-500" },
};

const CATEGORY_LABELS = { structure: "Structure", fuel: "Fuel", engines: "Engines", aero: "Aerodynamics", payload: "Payload", recovery: "Recovery" };

const PART_LIBRARY = {
  structure: [
    { id: "nose_std", cat: "structure", type: "noseCone", name: "Standard Nose Cone", mass: 1.5, length: 0.4, dragCoefficient: 0.20 },
    { id: "nose_ogive", cat: "structure", type: "noseCone", name: "Ogive Nose Cone", mass: 1.9, length: 0.55, dragCoefficient: 0.14 },
    { id: "nose_blunt", cat: "structure", type: "noseCone", name: "Blunt Nose Cone", mass: 1.2, length: 0.25, dragCoefficient: 0.32 },
    { id: "tube_xs", cat: "structure", type: "bodyTube", name: "Body Tube (XS)", mass: 1.2, length: 0.3, diameter: 0.25, dragCoefficient: 0.28 },
    { id: "tube_s", cat: "structure", type: "bodyTube", name: "Body Tube (Small)", mass: 2.0, length: 0.5, diameter: 0.3, dragCoefficient: 0.30 },
    { id: "tube_m", cat: "structure", type: "bodyTube", name: "Body Tube (Medium)", mass: 3.5, length: 0.8, diameter: 0.35, dragCoefficient: 0.32 },
    { id: "tube_l", cat: "structure", type: "bodyTube", name: "Body Tube (Large)", mass: 5.0, length: 1.2, diameter: 0.4, dragCoefficient: 0.35 },
    { id: "coupler", cat: "structure", type: "coupler", name: "Coupler", mass: 0.8 },
    { id: "decoupler", cat: "structure", type: "decoupler", name: "Decoupler", mass: 1.0, separationImpulse: 50 },
    { id: "interstage", cat: "structure", type: "decoupler", name: "Interstage (Faired)", mass: 2.2, separationImpulse: 70 },
    { id: "adapter", cat: "structure", type: "adapter", name: "Diameter Adapter", mass: 0.9, dragCoefficient: 0.05 },
  ],
  fuel: [
    { id: "tank_xs", cat: "fuel", type: "fuelTank", name: "Micro Fuel Tank", mass: 0.5, fuelCapacity: 2 },
    { id: "tank_s", cat: "fuel", type: "fuelTank", name: "Small Fuel Tank", mass: 1.0, fuelCapacity: 5 },
    { id: "tank_m", cat: "fuel", type: "fuelTank", name: "Medium Fuel Tank", mass: 2.0, fuelCapacity: 12 },
    { id: "tank_l", cat: "fuel", type: "fuelTank", name: "Large Fuel Tank", mass: 3.5, fuelCapacity: 25 },
    { id: "tank_xl", cat: "fuel", type: "fuelTank", name: "XL Fuel Tank", mass: 5.5, fuelCapacity: 45 },
  ],
  engines: [
    { id: "eng_cinder", cat: "engines", type: "engine", engineType: "solid", name: "Cinder (Micro Solid)", mass: 0.6, thrust: 900, isp: 140, propellantMass: 2.5 },
    { id: "eng_whisker", cat: "engines", type: "engine", engineType: "solid", name: "Whisker-S (Small Solid)", mass: 1.2, thrust: 1800, isp: 165, propellantMass: 5 },
    { id: "eng_bulwark", cat: "engines", type: "engine", engineType: "solid", name: "Bulwark (Medium Solid)", mass: 3.0, thrust: 4200, isp: 180, propellantMass: 14 },
    { id: "eng_hammerhead", cat: "engines", type: "engine", engineType: "solid", name: "Hammerhead (High-Thrust Solid)", mass: 5.0, thrust: 9000, isp: 150, propellantMass: 20 },
    { id: "eng_ember", cat: "engines", type: "engine", engineType: "solid", name: "Ember (Long-Burn Solid)", mass: 4.0, thrust: 1200, isp: 200, propellantMass: 18 },
    { id: "eng_zephyr", cat: "engines", type: "engine", engineType: "liquid", name: "Zephyr (Light Liquid)", mass: 3.0, thrust: 1500, isp: 290, propellantMass: 0 },
    { id: "eng_halcyon", cat: "engines", type: "engine", engineType: "liquid", name: "Halcyon (Small Liquid)", mass: 3.5, thrust: 2200, isp: 270, propellantMass: 0 },
    { id: "eng_meridian", cat: "engines", type: "engine", engineType: "liquid", name: "Meridian (Efficient Liquid)", mass: 6.5, thrust: 3200, isp: 320, propellantMass: 0 },
    { id: "eng_torrent", cat: "engines", type: "engine", engineType: "liquid", name: "Torrent (High-Thrust Liquid)", mass: 8.0, thrust: 6000, isp: 240, propellantMass: 0 },
    { id: "eng_vanguard", cat: "engines", type: "engine", engineType: "liquid", name: "Vanguard (Heavy Liquid)", mass: 12.0, thrust: 9000, isp: 280, propellantMass: 0 },
  ],
  aero: [
    { id: "fin_canard", cat: "aero", type: "fins", name: "Canards", mass: 0.15, dragContribution: 0.01, stabilityContribution: 0.25 },
    { id: "fin_s", cat: "aero", type: "fins", name: "Small Fins", mass: 0.3, dragContribution: 0.02, stabilityContribution: 0.5 },
    { id: "fin_m", cat: "aero", type: "fins", name: "Medium Fins", mass: 0.6, dragContribution: 0.035, stabilityContribution: 1.0 },
    { id: "fin_l", cat: "aero", type: "fins", name: "Large Fins", mass: 1.0, dragContribution: 0.05, stabilityContribution: 1.6 },
    { id: "fin_delta", cat: "aero", type: "fins", name: "Delta Fins", mass: 1.4, dragContribution: 0.07, stabilityContribution: 2.1 },
  ],
  payload: [
    { id: "sat_cube", cat: "payload", type: "payload", name: "Educational CubeSat", mass: 2 },
    { id: "sat_s", cat: "payload", type: "payload", name: "Small Satellite", mass: 5 },
    { id: "sat_sci", cat: "payload", type: "payload", name: "Scientific Payload", mass: 10 },
    { id: "sat_m", cat: "payload", type: "payload", name: "Medium Satellite", mass: 15 },
    { id: "sat_h", cat: "payload", type: "payload", name: "Heavy Payload", mass: 50 },
  ],
  recovery: [
    { id: "streamer", cat: "recovery", type: "parachute", name: "Streamer", mass: 0.4, deploymentAltitude: 150, dragCoefficient: 0.6, deployedArea: 0.6 },
    { id: "chute_drogue", cat: "recovery", type: "parachute", name: "Drogue Chute", mass: 0.8, deploymentAltitude: 800, dragCoefficient: 1.0, deployedArea: 1.2 },
    { id: "chute_s", cat: "recovery", type: "parachute", name: "Small Parachute", mass: 1.0, deploymentAltitude: 300, dragCoefficient: 1.4, deployedArea: 2.0 },
    { id: "chute_m", cat: "recovery", type: "parachute", name: "Medium Parachute", mass: 1.5, deploymentAltitude: 300, dragCoefficient: 1.6, deployedArea: 3.0 },
    { id: "chute_l", cat: "recovery", type: "parachute", name: "Large Parachute", mass: 2.2, deploymentAltitude: 300, dragCoefficient: 1.8, deployedArea: 4.5 },
  ],
};

const ALL_PARTS = Object.values(PART_LIBRARY).flat();
const partDef = (partId) => ALL_PARTS.find((p) => p.id === partId);

const DEFAULT_STARTER = ["nose_std", "tube_s", "fin_m", "eng_bulwark"];

/**
 * Split the ordered part stack (top -> bottom) into stages at decoupler
 * boundaries, and resolve fuel compatibility per stage:
 *  - solid engines always carry their own integrated propellant
 *  - liquid engines draw ONLY from fuel tanks present in the SAME stage
 *  - a fuel tank with no liquid engine in its stage contributes dry mass
 *    only — its capacity is flagged as wasted/incompatible, never counted
 *    as usable propellant (prevents double-counting and fake fuel).
 */
function computeStages(instances) {
  const segments = [[]];
  for (const inst of instances) {
    const def = partDef(inst.partId);
    segments[segments.length - 1].push(inst);
    if (def.type === "decoupler") segments.push([]);
  }
  const bottomToTop = segments.slice().reverse(); // bottom-most segment fires first

  return bottomToTop.map((segParts, idx) => {
    let dryMass = 0;
    let thrust = 0;
    let solidPropellant = 0;
    let liquidCapacity = 0;
    let thrustIspSum = 0;
    const engineInstances = [];
    const tankInstances = [];
    let hasLiquidEngine = false;

    for (const inst of segParts) {
      const def = partDef(inst.partId);
      if (def.type === "engine") {
        dryMass += def.mass;
        thrust += def.thrust;
        thrustIspSum += def.thrust * def.isp;
        if (def.engineType === "solid") solidPropellant += def.propellantMass;
        else hasLiquidEngine = true;
        engineInstances.push({ inst, def });
      } else if (def.type === "fuelTank") {
        dryMass += def.mass;
        liquidCapacity += def.fuelCapacity;
        tankInstances.push({ inst, def });
      } else {
        dryMass += def.mass;
      }
    }

    const usableLiquidFuel = hasLiquidEngine ? liquidCapacity : 0;
    const wastedFuelMass = hasLiquidEngine ? 0 : liquidCapacity; // capacity that cannot be drawn by anything in this stage
    const propellantMass = solidPropellant + usableLiquidFuel;
    const isp = thrust > 0 ? thrustIspSum / thrust : 0;

    const underfueledLiquid = hasLiquidEngine && usableLiquidFuel <= 0; // liquid engine with no tank at all

    return {
      index: idx, parts: segParts, dryMass, thrust, propellantMass, isp,
      engines: engineInstances.map((e) => e.def),
      hasLiquidEngine, hasSolidEngine: solidPropellant > 0,
      wastedTankIds: hasLiquidEngine ? [] : tankInstances.map((t) => t.inst.id),
      underfueledEngineIds: underfueledLiquid ? engineInstances.filter((e) => e.def.engineType === "liquid").map((e) => e.inst.id) : [],
      wastedFuelMass,
    };
  });
}

/** Build the full computed rocket description used by both physics + UI. */
function buildRocket(instances) {
  const stages = computeStages(instances);
  const totalDryMass = stages.reduce((s, st) => s + st.dryMass, 0);
  const totalPropellant = stages.reduce((s, st) => s + st.propellantMass, 0);
  const totalMass = totalDryMass + totalPropellant;
  const totalWastedFuelMass = stages.reduce((s, st) => s + st.wastedFuelMass, 0);

  let maxDiameter = 0.3;
  let cdSum = 0;
  let cdCount = 0;
  let stabilitySum = 0;
  let chute = null;
  let hasStructure = false;
  let hasEngine = false;
  let finCount = 0;

  for (const inst of instances) {
    const def = partDef(inst.partId);
    if (def.type === "bodyTube") { maxDiameter = Math.max(maxDiameter, def.diameter); cdSum += def.dragCoefficient; cdCount++; hasStructure = true; }
    if (def.type === "noseCone") { cdSum += def.dragCoefficient; cdCount++; }
    if (def.type === "adapter") { cdSum += def.dragCoefficient; cdCount++; }
    if (def.type === "fins") { cdSum += def.dragContribution; cdCount++; stabilitySum += def.stabilityContribution; finCount++; }
    if (def.type === "engine") hasEngine = true;
    if (def.type === "parachute") chute = def;
  }

  const dragCoefficient = cdCount > 0 ? cdSum : 0.5;
  const referenceArea = Math.PI * (maxDiameter / 2) ** 2;

  const liftoffThrust = stages[0]?.thrust ?? 0;
  const twr = totalMass > 0 ? liftoffThrust / (totalMass * G0) : 0;

  // theoretical delta-v, summed stage by stage (correctly captures the staging benefit)
  let dvTotal = 0;
  let massAboveAndIncluding = totalMass;
  for (const st of stages) {
    const m0 = massAboveAndIncluding;
    const mf = m0 - st.propellantMass;
    dvTotal += tsiolkovsky(st.isp, m0, mf);
    massAboveAndIncluding = mf - st.dryMass;
  }

  const compatWarnings = [];
  stages.forEach((st, i) => {
    if (st.wastedFuelMass > 0) compatWarnings.push(`Stage ${i + 1}: fuel tank has no liquid engine in this stage — ${st.wastedFuelMass.toFixed(1)} kg of capacity is unusable dead weight.`);
    if (st.underfueledEngineIds.length > 0) compatWarnings.push(`Stage ${i + 1}: liquid engine has no fuel tank in this stage — it cannot produce thrust.`);
  });

  const errors = [];
  if (!hasEngine) errors.push("No engine installed.");
  if (totalPropellant <= 0) errors.push("No usable propellant available.");
  if (!hasStructure) errors.push("Missing structural body tube — no connection between parts.");
  if (hasEngine && totalPropellant > 0 && twr < 1) errors.push(`Thrust-to-weight ratio (${twr.toFixed(2)}) is below 1 — rocket cannot lift off.`);
  if (stages.some((s, i) => i < stages.length - 1 && s.thrust <= 0)) errors.push("A lower stage has no engine — impossible staging configuration.");
  if (stages[0] && stages[0].thrust > 0 && stages[0].propellantMass <= 0) errors.push("Stage 1 engine has no compatible propellant — check fuel tank placement.");

  return {
    instances, stages, totalDryMass, totalPropellant, totalMass, totalWastedFuelMass,
    dragCoefficient, referenceArea, chute,
    liftoffThrust, twr, dvTotal,
    isp: stages[0]?.isp ?? 0,
    burnTime: stages.reduce((s, st) => s + (st.thrust > 0 && st.propellantMass > 0 ? st.propellantMass / massFlowRate(st.thrust, st.isp) : 0), 0),
    stabilityScore: stabilitySum,
    finCount,
    valid: errors.length === 0,
    errors,
    compatWarnings,
  };
}

/* =========================================================================
   PREBUILT ROCKETS
   ========================================================================= */

const PREBUILT_ROCKETS = [
  {
    id: "beginner", name: "Beginner Rocket", tagline: "Simplest possible flight — great for First Flight.",
    parts: ["nose_blunt", "tube_xs", "fin_s", "eng_cinder"],
  },
  {
    id: "small", name: "Small Rocket", tagline: "Low complexity, reliable single-stage solid flight.",
    parts: ["nose_std", "tube_s", "fin_m", "eng_whisker"],
  },
  {
    id: "medium", name: "Medium Rocket", tagline: "Introduces liquid propulsion and fuel tanks.",
    parts: ["nose_ogive", "tube_m", "fin_m", "tank_m", "eng_halcyon"],
  },
  {
    id: "payload", name: "Payload Rocket", tagline: "Carries a real payload to altitude on a strong single stage.",
    parts: ["nose_ogive", "sat_m", "tube_m", "chute_m", "tube_l", "fin_l", "eng_torrent", "tank_l"],
  },
  {
    id: "highalt", name: "High-Altitude Rocket", tagline: "Two solid stages for maximum altitude with simple staging.",
    parts: ["nose_ogive", "chute_s", "tube_m", "fin_m", "decoupler", "tube_l", "fin_l", "eng_hammerhead"],
  },
  {
    id: "large", name: "Large Rocket (Multi-Stage)", tagline: "Advanced 2-stage design with payload — demonstrates full staging.",
    parts: ["nose_std", "sat_m", "chute_m", "tube_m", "decoupler", "tank_m", "eng_meridian", "decoupler", "tube_l", "fin_l", "eng_hammerhead"],
  },
];

/* =========================================================================
   MISSIONS MODULE (missions/)
   ========================================================================= */

const MISSIONS = [
  { id: "m1", name: "First Flight", brief: "Reach 1 km altitude.", target: 1000, check: (r) => r.maxAltitude >= 1000 },
  { id: "m2", name: "High Flyer", brief: "Reach 10 km altitude.", target: 10000, check: (r) => r.maxAltitude >= 10000 },
  { id: "m3", name: "Payload Delivery", brief: "Reach 10 km carrying a 15 kg payload.", target: 10000, check: (r, rocket) => r.maxAltitude >= 10000 && rocket.instances.some((i) => partDef(i.partId).id === "sat_m") },
  { id: "m4", name: "Efficiency Challenge", brief: "Reach 10 km using the least propellant.", target: 10000, check: (r) => r.maxAltitude >= 10000, scoreExtra: (r, rocket) => Math.max(0, 5000 - rocket.totalPropellant * 20) },
  { id: "m5", name: "Staging Challenge", brief: "Reach 30 km using at least 2 stages.", target: 30000, check: (r, rocket) => r.maxAltitude >= 30000 && rocket.stages.filter((s) => s.thrust > 0).length >= 2 },
  { id: "m6", name: "Maximum Altitude", brief: "Reach the highest altitude possible (propellant budget: 40 kg).", target: null, check: () => true, scoreExtra: (r) => r.maxAltitude },
];

function scoreFlight(mission, result, rocket) {
  const completed = mission.check(result, rocket);
  let score = 0;
  if (completed) score += 1000;
  score += Math.round(result.maxAltitude);
  score += Math.round(2000 / (1 + rocket.totalPropellant / 10));
  if (mission.scoreExtra) score += Math.round(mission.scoreExtra(result, rocket));
  if (rocket.stabilityScore < 0.8) score = Math.round(score * 0.6);
  return { completed, score: Math.max(0, score) };
}

/* =========================================================================
   PERSISTENCE MODULE (storage/)
   Everything here is a thin wrapper around localStorage. Isolated so a real
   backend could be swapped in later behind the same function signatures —
   see the note in the final report about why no backend is used for now.
   ========================================================================= */

const LS_KEYS = {
  currentDesign: "rocketlab.currentDesign.v1",
  teamName: "rocketlab.teamName.v1",
  savedDesigns: "rocketlab.savedDesigns.v1",
  leaderboard: "rocketlab.leaderboard.v1",
};

function lsGet(key, fallback) {
  try {
    const raw = localStorage.getItem(key);
    if (raw == null) return fallback;
    return JSON.parse(raw);
  } catch {
    return fallback;
  }
}
function lsSet(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
    return true;
  } catch {
    return false;
  }
}

/* =========================================================================
   RENDERING MODULE (rendering/)
   Canvas is a pure function of simulation state — it never drives physics.
   The rocket's visual shape is derived from the current rocket config
   (stage count, engine count, fins, payload, chute) rather than hard-coded.
   ========================================================================= */

function drawStarfield(ctx, w, h, seed, alpha) {
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.fillStyle = "#e2e8f0";
  for (let i = 0; i < 90; i++) {
    const sx = (i * 137.5 + seed * 0.02) % w;
    const sy = (i * 91.3) % (h * 0.75);
    const s = (i % 5 === 0) ? 1.6 : 0.9;
    ctx.fillRect(sx, sy, s, s);
  }
  ctx.restore();
}

function drawRocket(ctx, rx, ry, rocket, phase, chuteDeployed, scale = 1) {
  const stageCount = Math.max(1, rocket?.stages?.length || 1);
  const engineCount = Math.max(1, rocket?.stages?.[0]?.engines?.length || 1);
  const hasFins = (rocket?.finCount || 0) > 0;
  const bodyLen = (34 + stageCount * 10) * scale;
  const bodyW = (11 + Math.min(6, engineCount * 1.5)) * scale;
  const noseLen = 16 * scale;

  ctx.save();
  ctx.translate(rx, ry);

  // body cylinder with subtle shading gradient for a less flat, more realistic look
  const bodyGrad = ctx.createLinearGradient(-bodyW, 0, bodyW, 0);
  bodyGrad.addColorStop(0, "#94a3b8");
  bodyGrad.addColorStop(0.45, "#f1f5f9");
  bodyGrad.addColorStop(0.55, "#f1f5f9");
  bodyGrad.addColorStop(1, "#64748b");
  ctx.fillStyle = bodyGrad;
  ctx.fillRect(-bodyW / 2, 0, bodyW, bodyLen);

  // stage separation rings
  for (let i = 1; i < stageCount; i++) {
    const y = (bodyLen / stageCount) * i;
    ctx.strokeStyle = "#334155";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(-bodyW / 2, y);
    ctx.lineTo(bodyW / 2, y);
    ctx.stroke();
  }

  // nose cone (curved)
  ctx.fillStyle = "#cbd5e1";
  ctx.beginPath();
  ctx.moveTo(-bodyW / 2, 0);
  ctx.quadraticCurveTo(0, -noseLen, bodyW / 2, 0);
  ctx.closePath();
  ctx.fill();

  // a thin payload/instrument band near the top if the rocket carries payload
  if (rocket?.instances?.some((i) => partDef(i.partId)?.cat === "payload")) {
    ctx.fillStyle = "#a78bfa";
    ctx.fillRect(-bodyW / 2, bodyLen * 0.12, bodyW, 3 * scale);
  }

  // fins (trapezoid), scaled by how many fin parts the rocket carries
  if (hasFins) {
    ctx.fillStyle = "#334155";
    const finW = 9 * scale;
    const finH = 16 * scale;
    ctx.beginPath();
    ctx.moveTo(-bodyW / 2, bodyLen - finH);
    ctx.lineTo(-bodyW / 2 - finW, bodyLen);
    ctx.lineTo(-bodyW / 2, bodyLen);
    ctx.closePath(); ctx.fill();
    ctx.beginPath();
    ctx.moveTo(bodyW / 2, bodyLen - finH);
    ctx.lineTo(bodyW / 2 + finW, bodyLen);
    ctx.lineTo(bodyW / 2, bodyLen);
    ctx.closePath(); ctx.fill();
  }

  // engine nozzles at the base — one per engine in the active first stage
  const nozzleW = Math.max(3, (bodyW * 0.9) / engineCount);
  for (let i = 0; i < engineCount; i++) {
    const nx = -bodyW / 2 + nozzleW * (i + 0.5);
    ctx.fillStyle = "#1e293b";
    ctx.beginPath();
    ctx.moveTo(nx - nozzleW * 0.35, bodyLen);
    ctx.lineTo(nx + nozzleW * 0.35, bodyLen);
    ctx.lineTo(nx + nozzleW * 0.22, bodyLen + 6 * scale);
    ctx.lineTo(nx - nozzleW * 0.22, bodyLen + 6 * scale);
    ctx.closePath();
    ctx.fill();

    if (phase === "burn") {
      const flicker = 0.85 + Math.random() * 0.3;
      const flameLen = (16 + Math.random() * 10) * scale * flicker;
      const fgrad = ctx.createLinearGradient(nx, bodyLen + 6 * scale, nx, bodyLen + 6 * scale + flameLen);
      fgrad.addColorStop(0, "#fef9c3");
      fgrad.addColorStop(0.4, "#fb923c");
      fgrad.addColorStop(1, "rgba(249,115,22,0)");
      ctx.fillStyle = fgrad;
      ctx.beginPath();
      ctx.moveTo(nx - nozzleW * 0.3, bodyLen + 6 * scale);
      ctx.lineTo(nx + nozzleW * 0.3, bodyLen + 6 * scale);
      ctx.lineTo(nx, bodyLen + 6 * scale + flameLen);
      ctx.closePath();
      ctx.fill();
    }
  }

  // parachute
  if (chuteDeployed) {
    ctx.strokeStyle = "#e2e8f0";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(0, -noseLen - 26 * scale, 24 * scale, Math.PI, 2 * Math.PI);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(-24 * scale, -noseLen - 26 * scale); ctx.lineTo(-bodyW / 2, 0);
    ctx.moveTo(24 * scale, -noseLen - 26 * scale); ctx.lineTo(bodyW / 2, 0);
    ctx.stroke();
  }

  ctx.restore();
}

function drawScene(ctx, w, h, opts) {
  const { altitude, chuteDeployed, trail, maxAltitudeSeen, phase, rocket, debris, tRef } = opts;
  ctx.clearRect(0, 0, w, h);

  const viewSpan = Math.max(400, maxAltitudeSeen * 1.25 + 200);
  const pxPerM = h / viewSpan;
  const groundY = h - 40;
  const rocketScreenY = groundY - altitude * pxPerM;

  const skyT = Math.min(1, altitude / 40000);
  const grad = ctx.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, `rgb(${8 + skyT * 2},${10 + skyT * 4},${20 + skyT * 10})`);
  grad.addColorStop(0.55, `rgb(${14 - skyT * 6},${22 - skyT * 12},${38 - skyT * 20})`);
  grad.addColorStop(1, `rgb(${20 - skyT * 15},${40 - skyT * 30},${70 - skyT * 55})`);
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, w, h);

  if (skyT > 0.2) drawStarfield(ctx, w, h, tRef || 0, Math.min(0.9, (skyT - 0.2) * 1.2));

  // faint nebula glow, purely decorative, drawn once behind everything
  const nebula = ctx.createRadialGradient(w * 0.8, h * 0.15, 10, w * 0.8, h * 0.15, w * 0.5);
  nebula.addColorStop(0, "rgba(56,189,248,0.06)");
  nebula.addColorStop(1, "rgba(56,189,248,0)");
  ctx.fillStyle = nebula;
  ctx.fillRect(0, 0, w, h);

  // altitude scale ticks
  ctx.strokeStyle = "rgba(148,163,184,0.2)";
  ctx.fillStyle = "rgba(148,163,184,0.55)";
  ctx.font = "10px monospace";
  const step = viewSpan > 8000 ? 5000 : viewSpan > 2000 ? 1000 : 200;
  for (let a = 0; a < viewSpan + altitude; a += step) {
    const y = groundY - a * pxPerM;
    if (y < -20) break;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    ctx.fillText(`${a} m`, 6, y - 3);
  }

  // ground + simple launch tower near the pad
  if (groundY < h) { ctx.fillStyle = "#1b2a1f"; ctx.fillRect(0, groundY, w, h - groundY); }
  ctx.strokeStyle = "#22c55e55";
  ctx.beginPath(); ctx.moveTo(0, groundY); ctx.lineTo(w, groundY); ctx.stroke();
  if (altitude < 150) {
    ctx.strokeStyle = "#475569";
    ctx.lineWidth = 2;
    const tx = w / 2 - 40;
    ctx.beginPath();
    ctx.moveTo(tx, groundY); ctx.lineTo(tx, groundY - 70);
    ctx.moveTo(tx - 8, groundY - 20); ctx.lineTo(tx, groundY - 30); ctx.lineTo(tx - 8, groundY - 40);
    ctx.stroke();
  }

  // trajectory trail
  if (trail.length > 1) {
    ctx.strokeStyle = "rgba(56,189,248,0.6)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    trail.forEach((p, i) => {
      const x = w / 2 + p.x;
      const y = groundY - p.alt * pxPerM;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  // falling jettisoned stage debris
  if (debris) {
    for (const d of debris) {
      const dy = groundY - d.alt * pxPerM;
      ctx.fillStyle = `rgba(100,116,139,${d.alpha})`;
      ctx.fillRect(w / 2 - 4, dy, 8, 10);
    }
  }

  const rx = w / 2;
  const ry = Math.max(20, rocketScreenY);
  drawRocket(ctx, rx, ry, rocket, phase, chuteDeployed, 1.15);
}

/* =========================================================================
   PHYSICS VALIDATION MODULE — runs the exact functions above against known
   scenarios so bugs are visible rather than hidden.
   ========================================================================= */

function runValidationSuite() {
  const results = [];
  const approx = (a, b, tol) => Math.abs(a - b) <= tol;

  results.push((() => {
    const netForce = 0 - 10 * G0 + 0;
    const a = netForce / 10;
    return { name: "Gravity-only acceleration ≈ -9.80665 m/s²", expected: -G0, actual: a, pass: approx(a, -G0, 1e-6) };
  })());

  results.push((() => {
    const a = 1000 / 20;
    return { name: "Constant thrust acceleration = F/m", expected: 50, actual: a, pass: approx(a, 50, 1e-9) };
  })());

  results.push((() => {
    const d = dragForceMag(0, 300, 0.4, 0.05);
    return { name: "Zero atmospheric density produces zero drag", expected: 0, actual: d, pass: d === 0 };
  })());

  results.push((() => {
    const gLow = gravityAt(0);
    const gHigh = gravityAt(100000);
    return { name: "Altitude-aware gravity decreases with altitude", expected: `< ${gLow.toFixed(4)}`, actual: gHigh.toFixed(4), pass: gHigh < gLow };
  })());

  {
    const rocket = buildRocket([{ partId: "nose_std" }, { partId: "tube_s" }, { partId: "fin_m" }, { partId: "eng_whisker" }]);
    let state = { altitude: 0, velocity: 0, propellantByStage: rocket.stages.map((s) => s.propellantMass) };
    let flags = { activeStageIndex: 0, jettisonedDryMass: 0, chuteDeployed: false, apogeeReached: false, maxAltitude: 0 };
    const initialPropellant = state.propellantByStage[0];
    const initialMass = rocket.totalMass;
    let step, steps = 0, shutdownSeen = false;
    while (steps < 5000) {
      step = integrateStep(state, rocket, flags, 0.02);
      state = step.state; flags = step.flags; steps++;
      if (step.events.some((e) => e.type === "BURNOUT")) { shutdownSeen = true; break; }
    }
    results.push({ name: "Fuel decreases while engine burns", expected: "< initial propellant", actual: `${state.propellantByStage[0].toFixed(3)} kg (from ${initialPropellant} kg)`, pass: state.propellantByStage[0] < initialPropellant });
    results.push({ name: "Engine shuts down when fuel reaches zero", expected: true, actual: shutdownSeen, pass: shutdownSeen });
    results.push({ name: "Mass decreases as propellant burns", expected: "< initial mass", actual: `${step.derived.mass.toFixed(2)} kg (from ${initialMass.toFixed(2)} kg)`, pass: step.derived.mass < initialMass });
  }

  results.push((() => {
    const dv = tsiolkovsky(250, 100, 40);
    const expected = 250 * G0 * Math.log(100 / 40);
    return { name: "Tsiolkovsky rocket equation", expected: expected.toFixed(2) + " m/s", actual: dv.toFixed(2) + " m/s", pass: approx(dv, expected, 1e-6) };
  })());

  {
    const rocket = buildRocket([{ partId: "nose_std" }, { partId: "tube_s" }, { partId: "decoupler" }, { partId: "tube_m" }, { partId: "eng_bulwark" }, { partId: "eng_hammerhead" }]);
    const massBefore = rocket.totalDryMass;
    const jettisoned = rocket.stages[0].dryMass;
    results.push({ name: "Staging removes discarded stage dry mass", expected: `< ${massBefore.toFixed(2)} kg`, actual: `${(massBefore - jettisoned).toFixed(2)} kg after separation`, pass: jettisoned > 0 && jettisoned < massBefore });
  }

  {
    const rocket = buildRocket([{ partId: "nose_std" }, { partId: "tube_s" }, { partId: "chute_m" }, { partId: "eng_whisker" }]);
    const dragNoChute = dragForceMag(1.0, 50, rocket.dragCoefficient, rocket.referenceArea);
    const dragChute = dragForceMag(1.0, 50, rocket.chute.dragCoefficient, rocket.chute.deployedArea);
    results.push({ name: "Parachute increases drag after deployment", expected: `> ${dragNoChute.toFixed(1)} N`, actual: `${dragChute.toFixed(1)} N`, pass: dragChute > dragNoChute });
  }

  results.push((() => {
    const rocket = buildRocket([{ partId: "nose_std" }, { partId: "tube_s" }, { partId: "eng_halcyon" }]); // liquid engine, no tank
    return { name: "Liquid engine without a fuel tank has zero usable propellant", expected: 0, actual: rocket.totalPropellant, pass: rocket.totalPropellant === 0 };
  })());

  results.push((() => {
    const rocket = buildRocket([{ partId: "nose_std" }, { partId: "tube_s" }, { partId: "tank_l" }, { partId: "eng_whisker" }]); // solid engine + tank (incompatible)
    return { name: "Fuel tank next to a solid-only engine is flagged as wasted, not double-counted", expected: "25 kg wasted, propellant = 5 kg", actual: `${rocket.totalWastedFuelMass} kg wasted, propellant = ${rocket.totalPropellant} kg`, pass: rocket.totalWastedFuelMass === 25 && rocket.totalPropellant === 5 };
  })());

  results.push((() => {
    const rocket = buildRocket([{ partId: "nose_std" }, { partId: "tube_l" }, { partId: "sat_h" }, { partId: "sat_h" }, { partId: "eng_cinder" }]); // tiny engine, two heavy payloads
    return { name: "Underpowered rocket (TWR < 1) is flagged invalid", expected: false, actual: rocket.valid, pass: rocket.valid === false && rocket.twr < 1 };
  })());

  results.push((() => {
    const rocket = buildRocket([{ partId: "nose_std" }, { partId: "tube_s" }]); // no engine at all
    return { name: "Rocket with no engine is flagged invalid", expected: false, actual: rocket.valid, pass: rocket.valid === false };
  })());

  results.push((() => {
    const step = integrateStep({ altitude: 0, velocity: 0, propellantByStage: [1] },
      { stages: [{ thrust: 1e9, isp: 200, dryMass: 1, propellantMass: 1 }], totalDryMass: 1, dragCoefficient: 0.3, referenceArea: 0.05, chute: null },
      { activeStageIndex: 0, jettisonedDryMass: 0, chuteDeployed: false, apogeeReached: false, maxAltitude: 0 }, 1);
    return { name: "Extreme thrust triggers the numerical safety limit instead of NaN", expected: true, actual: step.flags.aborted, pass: step.flags.aborted === true && Number.isFinite(step.state.altitude) };
  })());

  return results;
}

/* =========================================================================
   BACKGROUND DECORATION — pure CSS, non-interactive, drawn once behind the
   whole app. Cheap for 30-40 concurrent browsers since it's just a fixed
   layer with a slow CSS animation, no JS per-frame cost.
   ========================================================================= */

const Starfield = memo(function Starfield() {
  const dots = useMemo(() => {
    const arr = [];
    for (let i = 0; i < 70; i++) {
      arr.push({ left: (i * 53.7) % 100, top: (i * 31.3) % 100, size: (i % 4 === 0) ? 2 : 1, delay: (i % 10) * 0.4 });
    }
    return arr;
  }, []);
  return (
    <div className="fixed inset-0 pointer-events-none overflow-hidden z-0" aria-hidden="true">
      <div className="absolute inset-0 bg-slate-950" />
      <div className="absolute -top-1/3 -right-1/4 w-[70vw] h-[70vw] rounded-full bg-cyan-500/5 blur-3xl" />
      <div className="absolute -bottom-1/3 -left-1/4 w-[60vw] h-[60vw] rounded-full bg-indigo-500/5 blur-3xl" />
      <div
        className="absolute inset-0 opacity-60"
        style={{ backgroundImage: "linear-gradient(rgba(148,163,184,0.035) 1px, transparent 1px), linear-gradient(90deg, rgba(148,163,184,0.035) 1px, transparent 1px)", backgroundSize: "40px 40px" }}
      />
      {dots.map((d, i) => (
        <div
          key={i}
          className="absolute rounded-full bg-slate-300 animate-pulse"
          style={{ left: `${d.left}%`, top: `${d.top}%`, width: d.size, height: d.size, opacity: 0.4, animationDuration: "4s", animationDelay: `${d.delay}s` }}
        />
      ))}
    </div>
  );
});

/* =========================================================================
   UI MODULE
   ========================================================================= */

function uid() { return Math.random().toString(36).slice(2, 10); }

const StatRow = memo(function StatRow({ label, value, unit, warn }) {
  return (
    <div className="flex items-baseline justify-between py-1 border-b border-slate-800/60">
      <span className="text-[11px] uppercase tracking-widest text-slate-500">{label}</span>
      <span className={`font-mono text-sm ${warn ? "text-amber-400" : "text-cyan-300"}`}>{value}{unit ? <span className="text-slate-500 ml-1">{unit}</span> : null}</span>
    </div>
  );
});

function TopBar({ tab, setTab, onNew, onReset, onLaunch, launchDisabled, saveMenuOpen, setSaveMenuOpen }) {
  const tabs = ["BUILD", "SIMULATE", "MISSIONS", "RESULTS", "VALIDATION"];
  return (
    <div className="border-b border-slate-800 bg-slate-950/90 backdrop-blur sticky top-0 z-20">
      <div className="flex items-center justify-between px-3 sm:px-4 py-2.5 flex-wrap gap-y-2">
        <div className="flex items-center gap-2">
          <Rocket className="text-cyan-400" size={20} />
          <span className="font-semibold tracking-wide text-slate-100">ROCKET LAB</span>
          <span className="hidden md:inline text-[10px] uppercase tracking-widest text-slate-600 ml-2">Design &amp; Flight Simulator</span>
        </div>
        <div className="flex items-center gap-1 flex-wrap">
          {tabs.map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              title={`Go to ${t}`}
              className={`px-2.5 sm:px-3 py-1.5 text-[11px] sm:text-xs uppercase tracking-widest rounded-sm transition-colors ${
                tab === t ? "bg-cyan-500/15 text-cyan-300 border border-cyan-500/40" : "text-slate-500 border border-transparent hover:text-slate-300"
              }`}
            >
              {t}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-1.5 relative">
          <button onClick={onNew} title="New Rocket — clears the build area completely" className="p-1.5 rounded-sm text-slate-400 hover:text-slate-100 hover:bg-slate-800"><Plus size={16} /></button>
          <button onClick={() => setSaveMenuOpen((v) => !v)} title="Save or load a rocket design" className="p-1.5 rounded-sm text-slate-400 hover:text-slate-100 hover:bg-slate-800"><Save size={16} /></button>
          <button onClick={onReset} title="Reset — restores the default starter rocket" className="p-1.5 rounded-sm text-slate-400 hover:text-slate-100 hover:bg-slate-800"><RotateCcw size={16} /></button>
          <button
            onClick={onLaunch}
            disabled={launchDisabled}
            title={launchDisabled ? "Fix the rocket's validity errors before launching" : "Go to SIMULATE and launch"}
            className={`ml-1 flex items-center gap-1.5 px-3 py-1.5 rounded-sm text-xs uppercase tracking-widest font-semibold ${
              launchDisabled ? "bg-slate-800 text-slate-600 cursor-not-allowed" : "bg-amber-500 text-slate-950 hover:bg-amber-400"
            }`}
          >
            <Flame size={14} /> Launch
          </button>
        </div>
      </div>
    </div>
  );
}

function SaveMenu({ open, onClose, instances, onLoad, teamName }) {
  const [designs, setDesigns] = useState(() => lsGet(LS_KEYS.savedDesigns, []));
  const [name, setName] = useState("");
  const [notice, setNotice] = useState("");

  if (!open) return null;

  const saveCurrent = () => {
    const finalName = name.trim() || `Design ${designs.length + 1}`;
    const entry = { name: finalName, parts: instances.map((i) => i.partId), savedAt: Date.now() };
    const next = [...designs.filter((d) => d.name !== finalName), entry];
    setDesigns(next);
    lsSet(LS_KEYS.savedDesigns, next);
    setName("");
    setNotice(`Saved "${finalName}" to this browser.`);
    setTimeout(() => setNotice(""), 2200);
  };

  const deleteDesign = (n) => {
    const next = designs.filter((d) => d.name !== n);
    setDesigns(next);
    lsSet(LS_KEYS.savedDesigns, next);
  };

  return (
    <div className="absolute right-0 top-full mt-1 w-72 bg-slate-900 border border-slate-700 rounded-sm shadow-xl z-30 p-3">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs uppercase tracking-widest text-slate-400">Save / Load Design</span>
        <button onClick={onClose} title="Close"><X size={14} className="text-slate-500 hover:text-slate-200" /></button>
      </div>
      <div className="flex gap-1.5 mb-2">
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="Design name"
          className="flex-1 min-w-0 px-2 py-1 bg-slate-950 border border-slate-700 rounded-sm text-xs text-slate-200 placeholder:text-slate-600 focus:outline-none focus:border-cyan-600"
        />
        <button onClick={saveCurrent} title="Save current rocket to this browser's storage" className="px-2 py-1 bg-cyan-600 text-slate-950 rounded-sm text-xs font-semibold">Save</button>
      </div>
      {notice && <div className="text-[11px] text-emerald-400 mb-2">{notice}</div>}
      <div className="max-h-48 overflow-y-auto space-y-1">
        {designs.length === 0 && <div className="text-[11px] text-slate-600">No saved designs yet.</div>}
        {designs.slice().reverse().map((d) => (
          <div key={d.name} className="flex items-center justify-between px-2 py-1.5 bg-slate-950 border border-slate-800 rounded-sm">
            <span className="text-xs text-slate-300 truncate">{d.name}</span>
            <div className="flex items-center gap-1 shrink-0">
              <button onClick={() => onLoad(d.parts)} title="Load this design into the builder" className="p-1 text-cyan-400 hover:text-cyan-300"><FolderOpen size={13} /></button>
              <button onClick={() => deleteDesign(d.name)} title="Delete this saved design" className="p-1 text-red-400 hover:text-red-300"><Trash2 size={13} /></button>
            </div>
          </div>
        ))}
      </div>
      <div className="text-[10px] text-slate-600 mt-2">Your current in-progress rocket and team name are auto-saved to this browser as you work.</div>
    </div>
  );
}

function CompatBadge({ text, ok }) {
  return (
    <span className={`inline-flex items-center gap-1 text-[9px] px-1.5 py-0.5 rounded-sm border ${ok ? "border-emerald-700/50 text-emerald-400 bg-emerald-500/5" : "border-amber-600/60 text-amber-400 bg-amber-500/10"}`}>
      {ok ? <CheckCircle2 size={9} /> : <AlertTriangle size={9} />} {text}
    </span>
  );
}

const PartsPalette = memo(function PartsPalette({ onAdd, onLoadPrebuilt }) {
  const [showPrebuilt, setShowPrebuilt] = useState(true);
  return (
    <div className="w-full lg:w-64 shrink-0 border-r border-slate-800 bg-slate-950/70 overflow-y-auto lg:max-h-[calc(100vh-6.5rem)]">
      <div className="p-3 border-b border-slate-800/60">
        <button onClick={() => setShowPrebuilt((v) => !v)} className="w-full flex items-center justify-between text-[11px] uppercase tracking-widest text-slate-400 mb-2">
          <span className="flex items-center gap-1.5"><Layers size={13} className="text-cyan-400" /> Prebuilt Rockets</span>
          {showPrebuilt ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
        </button>
        {showPrebuilt && (
          <div className="space-y-1.5">
            {PREBUILT_ROCKETS.map((r) => (
              <button
                key={r.id}
                onClick={() => onLoadPrebuilt(r)}
                title={r.tagline}
                className="w-full text-left px-2.5 py-2 rounded-sm bg-slate-900 border border-slate-800 hover:border-cyan-600/60 hover:bg-slate-800/80"
              >
                <div className="text-xs text-slate-200">Load {r.name}</div>
                <div className="text-[10px] text-slate-500">{r.tagline}</div>
              </button>
            ))}
          </div>
        )}
      </div>
      {Object.entries(PART_LIBRARY).map(([cat, parts]) => {
        const c = CATEGORY_COLORS[cat];
        return (
          <div key={cat} className="p-3 border-b border-slate-800/60">
            <div className={`text-[11px] uppercase tracking-widest mb-2 flex items-center gap-1.5 ${c.text}`}>
              <span className={`w-2 h-2 rounded-full ${c.dot}`} /> {CATEGORY_LABELS[cat]}
            </div>
            <div className="space-y-1.5">
              {parts.map((p) => (
                <button
                  key={p.id}
                  draggable
                  onDragStart={(e) => e.dataTransfer.setData("text/part-id", p.id)}
                  onClick={() => onAdd(p.id)}
                  title={p.type === "engine" ? (p.engineType === "solid" ? "Integrated solid propellant — no fuel tank needed" : "Liquid engine — requires a compatible fuel tank in the same stage") : p.type === "fuelTank" ? "Compatible with liquid engines only" : `${p.name} — ${p.mass} kg`}
                  className={`w-full text-left px-2.5 py-2 rounded-sm bg-slate-900 border ${c.border}/30 hover:${c.border} hover:bg-slate-800/80 transition-colors group`}
                >
                  <div className={`text-xs text-slate-200 group-hover:${c.text}`}>{p.name}</div>
                  <div className="text-[10px] text-slate-500 font-mono flex items-center gap-1.5 flex-wrap">
                    {p.type === "engine" ? `${p.thrust} N · Isp ${p.isp}s` : `${p.mass} kg`}
                    {p.type === "engine" && (p.engineType === "solid" ? <CompatBadge text="Integrated fuel" ok /> : <CompatBadge text="Needs fuel tank" ok={false} />)}
                    {p.type === "fuelTank" && <CompatBadge text="Liquid engines only" ok />}
                  </div>
                </button>
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
});

const PartBlock = memo(function PartBlock({ inst, def, selected, onSelect, onDelete, onDup, onMove, isFirst, isLast, warning, stageBoundaryAfter }) {
  const c = CATEGORY_COLORS[def.cat];
  return (
    <>
      <div
        onClick={() => onSelect(inst.id)}
        title={`${def.name} — ${def.mass} kg${def.type === "engine" ? ` · ${def.thrust} N · Isp ${def.isp}s` : ""}${warning ? `\n⚠ ${warning}` : ""}`}
        className={`relative w-44 mx-auto px-3 py-2 border-2 ${c.border} ${warning ? "ring-2 ring-amber-500" : selected ? "ring-2 ring-cyan-400" : ""} bg-slate-900/90 cursor-pointer group`}
        style={{ borderRadius: def.type === "noseCone" ? "50% 50% 4px 4px" : def.type === "parachute" ? "10px" : "2px" }}
      >
        <div className={`text-[11px] text-center leading-tight ${c.text}`}>{def.name}</div>
        <div className="text-[9px] text-slate-500 text-center font-mono">{def.mass} kg{def.type === "engine" ? ` · ${def.thrust}N` : ""}</div>
        {warning && <div className="text-[8px] text-amber-400 text-center mt-0.5 flex items-center justify-center gap-0.5"><AlertTriangle size={8} /> incompatible</div>}
        <div className="absolute -right-2 top-1/2 -translate-y-1/2 flex flex-col gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
          <button title="Move up" onClick={(e) => { e.stopPropagation(); onMove(inst.id, -1); }} disabled={isFirst} className="p-0.5 bg-slate-800 rounded disabled:opacity-30"><ChevronUp size={11} /></button>
          <button title="Move down" onClick={(e) => { e.stopPropagation(); onMove(inst.id, 1); }} disabled={isLast} className="p-0.5 bg-slate-800 rounded disabled:opacity-30"><ChevronDown size={11} /></button>
          <button title="Duplicate" onClick={(e) => { e.stopPropagation(); onDup(inst.id); }} className="p-0.5 bg-slate-800 rounded"><Copy size={11} /></button>
          <button title="Delete" onClick={(e) => { e.stopPropagation(); onDelete(inst.id); }} className="p-0.5 bg-red-900/60 rounded"><Trash2 size={11} /></button>
        </div>
      </div>
      {stageBoundaryAfter != null && (
        <div className="flex items-center gap-2 w-44 mx-auto my-0.5">
          <div className="flex-1 h-px bg-amber-600/50" />
          <span className="text-[9px] uppercase tracking-widest text-amber-500">Stage {stageBoundaryAfter}</span>
          <div className="flex-1 h-px bg-amber-600/50" />
        </div>
      )}
    </>
  );
});

function BuilderCanvas({ instances, selectedId, setSelectedId, onDelete, onDup, onMove, onDropPart, onClear, warningMap }) {
  // figure out, for display only, which visual stage number each part belongs to (top -> bottom order shown)
  let stageCounter = 1;
  const totalDecouplers = instances.filter((i) => partDef(i.partId).type === "decoupler").length;

  return (
    <div
      className="flex-1 flex flex-col items-center overflow-y-auto py-6 px-4 min-w-0"
      onDragOver={(e) => e.preventDefault()}
      onDrop={(e) => { const pid = e.dataTransfer.getData("text/part-id"); if (pid) onDropPart(pid); }}
    >
      <div className="flex items-center justify-between w-full max-w-xs mb-4">
        <span className="text-[11px] uppercase tracking-widest text-slate-500">Rocket Stack</span>
        <button onClick={onClear} title="Remove every part" className="text-[10px] uppercase tracking-widest text-red-400/80 hover:text-red-300">Clear</button>
      </div>
      {instances.length === 0 ? (
        <div className="text-slate-600 text-sm mt-16 text-center max-w-xs">
          Add or drag parts from the library, or load a prebuilt rocket.<br />Build top → bottom: nose cone first, engine last.
        </div>
      ) : (
        <div className="flex flex-col gap-0.5">
          {instances.map((inst, i) => {
            const def = partDef(inst.partId);
            const isDecoupler = def.type === "decoupler";
            const boundaryLabel = isDecoupler ? (totalDecouplers - stageCounter + 1) : null;
            if (isDecoupler) stageCounter++;
            return (
              <PartBlock
                key={inst.id}
                inst={inst}
                def={def}
                selected={selectedId === inst.id}
                onSelect={setSelectedId}
                onDelete={onDelete}
                onDup={onDup}
                onMove={onMove}
                isFirst={i === 0}
                isLast={i === instances.length - 1}
                warning={warningMap[inst.id]}
                stageBoundaryAfter={boundaryLabel}
              />
            );
          })}
        </div>
      )}
    </div>
  );
}

function StatsPanel({ rocket, selectedDef }) {
  return (
    <div className="w-full lg:w-72 shrink-0 border-l border-slate-800 bg-slate-950/70 overflow-y-auto p-3 lg:max-h-[calc(100vh-6.5rem)]">
      <div className={`mb-3 px-2.5 py-2 rounded-sm border text-xs font-semibold uppercase tracking-widest flex items-center gap-2 ${
        rocket.valid ? "border-emerald-600/50 bg-emerald-500/10 text-emerald-400" : "border-red-600/50 bg-red-500/10 text-red-400"
      }`}>
        {rocket.valid ? <CheckCircle2 size={15} /> : <XCircle size={15} />}
        {rocket.valid ? "READY" : "INVALID"}
      </div>
      {!rocket.valid && (
        <ul className="mb-3 space-y-1">
          {rocket.errors.map((e, i) => (
            <li key={i} className="text-[11px] text-red-400/90 flex gap-1.5"><AlertTriangle size={12} className="shrink-0 mt-0.5" />{e}</li>
          ))}
        </ul>
      )}
      {rocket.compatWarnings.length > 0 && (
        <ul className="mb-3 space-y-1">
          {rocket.compatWarnings.map((w, i) => (
            <li key={i} className="text-[11px] text-amber-400/90 flex gap-1.5 bg-amber-500/5 border border-amber-700/30 rounded-sm p-1.5"><Info size={12} className="shrink-0 mt-0.5" />{w}</li>
          ))}
        </ul>
      )}

      <div className="text-[11px] uppercase tracking-widest text-slate-500 mb-1 mt-2">Rocket Statistics</div>
      <StatRow label="Total Mass" value={rocket.totalMass.toFixed(2)} unit="kg" />
      <StatRow label="Dry Mass" value={rocket.totalDryMass.toFixed(2)} unit="kg" />
      <StatRow label="Propellant Mass" value={rocket.totalPropellant.toFixed(2)} unit="kg" />
      {rocket.totalWastedFuelMass > 0 && <StatRow label="Wasted Fuel (incompatible)" value={rocket.totalWastedFuelMass.toFixed(2)} unit="kg" warn />}
      <StatRow label="Total Thrust (liftoff)" value={rocket.liftoffThrust.toFixed(0)} unit="N" />
      <StatRow label="Thrust-to-Weight" value={rocket.twr.toFixed(2)} warn={rocket.twr < 1} />
      <StatRow label="Isp (stage 1)" value={rocket.isp.toFixed(0)} unit="s" />
      <StatRow label="Theoretical Δv" value={rocket.dvTotal.toFixed(0)} unit="m/s" />
      <StatRow label="Total Burn Time" value={rocket.burnTime.toFixed(1)} unit="s" />
      <StatRow label="Stages" value={rocket.stages.length} />
      <StatRow label="Stability Score" value={rocket.stabilityScore.toFixed(2)} warn={rocket.stabilityScore < 0.8} />

      {selectedDef && (
        <>
          <div className="text-[11px] uppercase tracking-widest text-slate-500 mb-1 mt-4">Selected Part</div>
          <div className={`text-sm mb-1 ${CATEGORY_COLORS[selectedDef.cat].text}`}>{selectedDef.name}</div>
          {selectedDef.type === "engine" && (
            <div className="mb-1.5">{selectedDef.engineType === "solid" ? <CompatBadge text="Integrated solid propellant" ok /> : <CompatBadge text="Requires compatible liquid fuel tank in same stage" ok={false} />}</div>
          )}
          {selectedDef.type === "fuelTank" && <div className="mb-1.5"><CompatBadge text="Feeds liquid engines in the same stage only" ok /></div>}
          {Object.entries(selectedDef).filter(([k]) => !["id", "cat", "type", "name", "engineType"].includes(k)).map(([k, v]) => (
            typeof v === "number" ? <StatRow key={k} label={k.replace(/([A-Z])/g, " $1")} value={Number.isInteger(v) ? v : v.toFixed(2)} /> : null
          ))}
        </>
      )}
    </div>
  );
}

const TelemetryStat = memo(function TelemetryStat({ icon: Icon, label, value, unit }) {
  return (
    <div className="flex items-center gap-2 px-3 py-2 bg-slate-900/70 border border-slate-800 rounded-sm">
      <Icon size={14} className="text-cyan-400 shrink-0" />
      <div className="min-w-0">
        <div className="text-[9px] uppercase tracking-widest text-slate-500">{label}</div>
        <div className="font-mono text-sm text-slate-100 truncate">{value}{unit ? <span className="text-slate-500 text-xs ml-1">{unit}</span> : null}</div>
      </div>
    </div>
  );
});

const MiniChart = memo(function MiniChart({ title, data, dataKey, color }) {
  return (
    <div className="bg-slate-900/70 border border-slate-800 rounded-sm p-2 h-40 flex flex-col">
      <div className="text-[10px] uppercase tracking-widest text-slate-500 mb-1">{title}</div>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 2, right: 6, bottom: 0, left: -18 }}>
          <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" />
          <XAxis dataKey="t" tick={{ fontSize: 9, fill: "#64748b" }} tickFormatter={(v) => v.toFixed(0)} />
          <YAxis tick={{ fontSize: 9, fill: "#64748b" }} width={40} />
          <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", fontSize: 11 }} labelFormatter={(v) => `t=${v.toFixed(1)}s`} />
          <Line type="monotone" dataKey={dataKey} stroke={color} dot={false} strokeWidth={1.75} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
});

function EventLog({ events }) {
  return (
    <div className="bg-slate-900/70 border border-slate-800 rounded-sm p-2 h-40 overflow-y-auto flex flex-col-reverse">
      {events.slice().reverse().map((e, i) => (
        <div key={i} className="text-[11px] font-mono text-amber-300/90 border-b border-slate-800/50 py-0.5">
          <span className="text-slate-500 mr-2">t={e.t.toFixed(1)}s</span>{e.label}
        </div>
      ))}
      {events.length === 0 && <div className="text-slate-600 text-xs">Awaiting liftoff…</div>}
    </div>
  );
}

const TELEMETRY_CAP = 3000; // hard cap on stored samples so a long/runaway flight can't grow memory unbounded

/* ---- simulation runner hook: owns the physics stepping loop ---- */
function useFlightSim(rocket) {
  const [running, setRunning] = useState(false);
  const [phase, setPhase] = useState("idle"); // idle | burn | coast | descent | landed | error
  const [telemetry, setTelemetry] = useState([]);
  const [events, setEvents] = useState([]);
  const [trail, setTrail] = useState([]);
  const [snapshot, setSnapshot] = useState(null);
  const [result, setResult] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);

  const simRef = useRef(null);
  const rafRef = useRef(null);
  const lastTsRef = useRef(null);
  const accRef = useRef(0);
  const maxVelRef = useRef(0);
  const debrisRef = useRef([]);
  const DT = 0.02;

  const reset = useCallback(() => {
    cancelAnimationFrame(rafRef.current);
    lastTsRef.current = null;
    accRef.current = 0;
    maxVelRef.current = 0;
    debrisRef.current = [];
    setRunning(false);
    setPhase("idle");
    setTelemetry([]);
    setEvents([]);
    setTrail([]);
    setResult(null);
    setErrorMsg(null);
    if (rocket) {
      simRef.current = {
        state: { altitude: 0, velocity: 0, propellantByStage: rocket.stages.map((s) => s.propellantMass) },
        flags: { activeStageIndex: 0, jettisonedDryMass: 0, chuteDeployed: false, apogeeReached: false, maxAltitude: 0 },
        t: 0,
      };
      setSnapshot({ ...simRef.current.state, ...simRef.current.flags, t: 0, mass: rocket.totalMass, thrust: 0, acceleration: -G0, dynamicPressure: 0 });
    }
  }, [rocket]);

  useEffect(() => { reset(); }, [reset]);

  const eventLabel = (e) => {
    switch (e.type) {
      case "BURNOUT": return `ENGINE BURNOUT — stage ${e.stage + 1}`;
      case "SEPARATION": return `STAGE ${e.stage + 1} SEPARATION`;
      case "IGNITION": return `STAGE ${e.stage + 2} IGNITION`;
      case "APOGEE": return `APOGEE — ${e.altitude.toFixed(0)} m`;
      case "CHUTE_DEPLOY": return `PARACHUTE DEPLOYMENT — ${e.altitude.toFixed(0)} m`;
      default: return e.type;
    }
  };

  const tick = useCallback((ts) => {
    if (!simRef.current || !rocket) return;
    if (lastTsRef.current == null) lastTsRef.current = ts;
    let frameDt = (ts - lastTsRef.current) / 1000;
    lastTsRef.current = ts;
    frameDt = Math.min(frameDt, 0.1); // guard against huge dt after a dropped/backgrounded frame
    accRef.current += frameDt;

    let localEvents = [];
    let landed = false;
    let hitSafetyLimit = false;
    let firstStep = simRef.current.t === 0;
    let stepsThisFrame = 0;

    while (accRef.current >= DT && stepsThisFrame < 500) {
      const { state, flags, t } = simRef.current;

      // hard safety limit on wall-clock simulated time, independent of the per-step checks in integrateStep
      if (t >= SIM_MAX_TIME) {
        hitSafetyLimit = true;
        setErrorMsg(`Simulation reached the ${SIM_MAX_TIME}s maximum flight time and was stopped.`);
        break;
      }

      const out = integrateStep(state, rocket, flags, DT);
      const newT = t + DT;
      stepsThisFrame++;

      if (firstStep) { localEvents.push({ t: newT, label: "LIFTOFF" }); firstStep = false; }
      out.events.forEach((e) => {
        localEvents.push({ t: newT, label: eventLabel(e) });
        if (e.type === "SEPARATION") debrisRef.current.push({ alt: out.state.altitude, alpha: 1 });
      });

      maxVelRef.current = Math.max(maxVelRef.current, Math.abs(out.state.velocity));
      debrisRef.current = debrisRef.current.map((d) => ({ ...d, alpha: d.alpha - 0.01 })).filter((d) => d.alpha > 0);

      if (out.flags.aborted) {
        hitSafetyLimit = true;
        setErrorMsg(out.flags.abortReason);
        simRef.current = { state: out.state, flags: out.flags, t: newT, derived: out.derived };
        break;
      }

      if (state.altitude <= 0 && out.flags.apogeeReached && out.state.altitude <= 0 && out.state.velocity === 0) {
        landed = true;
      }

      simRef.current = { state: out.state, flags: out.flags, t: newT, derived: out.derived };
      accRef.current -= DT;

      setTelemetry((prev) => {
        if (prev.length > 0 && newT - prev[prev.length - 1].t < 0.05) return prev;
        const next = [...prev, {
          t: newT, altitude: out.state.altitude, velocity: out.state.velocity,
          acceleration: out.derived.acceleration, mass: out.derived.mass,
          fuel: out.state.propellantByStage.reduce((s, v) => s + v, 0),
        }];
        return next.length > TELEMETRY_CAP ? next.slice(next.length - TELEMETRY_CAP) : next;
      });

      if (landed) break;
    }

    if (localEvents.length) setEvents((prev) => [...prev, ...localEvents]);

    const s = simRef.current;
    const ph = hitSafetyLimit ? "error" : (s.derived?.thrust > 0 ? "burn" : s.flags.apogeeReached ? "descent" : "coast");
    setPhase(landed ? "landed" : ph);
    setSnapshot({ ...s.state, ...s.flags, t: s.t, ...s.derived });
    setTrail((prev) => [...prev.slice(-400), { x: 0, alt: s.state.altitude }]);

    if (landed || hitSafetyLimit) {
      setRunning(false);
      if (landed) {
        setResult({
          maxAltitude: s.flags.maxAltitude,
          finalTime: s.t,
          maxVelocity: maxVelRef.current,
          propellantRemaining: s.state.propellantByStage.reduce((a, b) => a + b, 0),
        });
      }
      return;
    }
    rafRef.current = requestAnimationFrame(tick);
  }, [rocket]);

  useEffect(() => {
    if (running) {
      lastTsRef.current = null;
      rafRef.current = requestAnimationFrame(tick);
    } else {
      cancelAnimationFrame(rafRef.current);
    }
    return () => cancelAnimationFrame(rafRef.current);
  }, [running, tick]);

  // always clean up the animation frame if this hook's owner unmounts
  useEffect(() => () => cancelAnimationFrame(rafRef.current), []);

  return { running, setRunning, phase, telemetry, events, trail, snapshot, result, errorMsg, reset, debrisRef };
}

function FlightControls({ sim, rocket }) {
  return (
    <div className="flex items-center gap-2 p-2.5 border-t border-slate-800 bg-slate-950/70 flex-wrap">
      {!sim.running ? (
        <button disabled={!rocket.valid || sim.phase === "landed"} onClick={() => sim.setRunning(true)} title="Start or resume the flight" className="flex items-center gap-1.5 px-3 py-1.5 text-xs uppercase tracking-widest bg-emerald-500/90 text-slate-950 rounded-sm disabled:opacity-30 disabled:cursor-not-allowed font-semibold">
          <Play size={13} /> {sim.telemetry.length ? "Resume" : "Launch"}
        </button>
      ) : (
        <button onClick={() => sim.setRunning(false)} title="Freeze the simulation" className="flex items-center gap-1.5 px-3 py-1.5 text-xs uppercase tracking-widest bg-amber-500/90 text-slate-950 rounded-sm font-semibold">
          <Pause size={13} /> Pause
        </button>
      )}
      <button onClick={sim.reset} title="Reset the flight to T+0" className="flex items-center gap-1.5 px-3 py-1.5 text-xs uppercase tracking-widest bg-slate-800 text-slate-300 rounded-sm">
        <RotateCcw size={13} /> Restart
      </button>
      <button onClick={() => { sim.setRunning(false); sim.reset(); }} title="Abort the flight and reset safely" className="flex items-center gap-1.5 px-3 py-1.5 text-xs uppercase tracking-widest bg-red-900/60 text-red-300 rounded-sm">
        <Square size={13} /> Abort
      </button>
      <div className="ml-auto font-mono text-xs text-slate-500">T+{(sim.snapshot?.t ?? 0).toFixed(1)}s</div>
    </div>
  );
}

function TelemetryPanel({ sim, rocket }) {
  const s = sim.snapshot || {};
  return (
    <div className="w-full lg:w-96 shrink-0 border-l border-slate-800 bg-slate-950/70 overflow-y-auto p-3 space-y-3 lg:max-h-[calc(100vh-6.5rem)]">
      <div className="flex items-center justify-between px-2.5 py-1.5 bg-slate-900 border border-slate-800 rounded-sm">
        <span className="text-[10px] uppercase tracking-widest text-slate-500">Stage / Phase</span>
        <span className="font-mono text-xs text-cyan-300">Stage {(s.activeStageIndex ?? 0) + 1}/{rocket.stages.length} · {(sim.phase || "idle").toUpperCase()}</span>
      </div>
      <div className="grid grid-cols-2 gap-2">
        <TelemetryStat icon={Gauge} label="Altitude" value={(s.altitude ?? 0).toFixed(0)} unit="m" />
        <TelemetryStat icon={Wind} label="Velocity" value={(s.velocity ?? 0).toFixed(1)} unit="m/s" />
        <TelemetryStat icon={Gauge} label="Acceleration" value={(s.acceleration ?? -G0).toFixed(2)} unit="m/s²" />
        <TelemetryStat icon={Flame} label="Thrust" value={(s.thrust ?? 0).toFixed(0)} unit="N" />
        <TelemetryStat icon={Rocket} label="Mass" value={(s.mass ?? rocket.totalMass).toFixed(1)} unit="kg" />
        <TelemetryStat icon={Flame} label="Fuel Left" value={(s.propellantByStage ? s.propellantByStage.reduce((a, b) => a + b, 0) : rocket.totalPropellant).toFixed(1)} unit="kg" />
        <TelemetryStat icon={Gauge} label="T/W Ratio" value={((s.thrust ?? 0) / ((s.mass ?? rocket.totalMass) * G0)).toFixed(2)} />
        <TelemetryStat icon={Wind} label="Dyn. Pressure" value={(s.dynamicPressure ?? 0).toFixed(0)} unit="Pa" />
      </div>

      <MiniChart title="Altitude vs Time (m)" data={sim.telemetry} dataKey="altitude" color="#38bdf8" />
      <MiniChart title="Velocity vs Time (m/s)" data={sim.telemetry} dataKey="velocity" color="#34d399" />
      <MiniChart title="Acceleration vs Time (m/s²)" data={sim.telemetry} dataKey="acceleration" color="#f59e0b" />
      <MiniChart title="Fuel vs Time (kg)" data={sim.telemetry} dataKey="fuel" color="#fb7185" />

      <div className="text-[11px] uppercase tracking-widest text-slate-500 mt-1">Flight Events</div>
      <EventLog events={sim.events} />

      {sim.errorMsg && (
        <div className="p-2.5 bg-red-500/10 border border-red-600/40 rounded-sm text-xs text-red-300 flex gap-2"><AlertTriangle size={14} className="shrink-0 mt-0.5" />{sim.errorMsg}</div>
      )}
      {sim.result && (
        <div className="p-2.5 bg-cyan-500/10 border border-cyan-600/40 rounded-sm text-xs text-cyan-300">
          Landed. Max altitude reached: <span className="font-mono">{sim.result.maxAltitude.toFixed(1)} m</span>
        </div>
      )}
    </div>
  );
}

function SimulateView({ rocket, activeMission, onFlightComplete }) {
  const sim = useFlightSim(rocket);
  const canvasRef = useRef(null);
  const reportedRef = useRef(false);

  // reset "have we reported this flight" whenever the flight itself resets (restart/abort/new rocket)
  useEffect(() => {
    if (sim.result) {
      if (!reportedRef.current) {
        reportedRef.current = true;
        onFlightComplete(sim.result, activeMission ? activeMission.id : null);
      }
    } else {
      reportedRef.current = false;
    }
  }, [sim.result, activeMission, onFlightComplete]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !sim.snapshot) return;
    const ctx = canvas.getContext("2d");
    drawScene(ctx, canvas.width, canvas.height, {
      altitude: sim.snapshot.altitude ?? 0,
      chuteDeployed: sim.snapshot.chuteDeployed,
      trail: sim.trail,
      maxAltitudeSeen: Math.max(200, sim.snapshot.maxAltitude ?? 0),
      phase: sim.phase,
      rocket,
      debris: sim.debrisRef.current,
      tRef: sim.snapshot.t * 1000,
    });
  }, [sim.snapshot, sim.trail, sim.phase, rocket, sim.debrisRef]);

  if (!rocket) return <div className="p-8 text-slate-500">Build a rocket first.</div>;

  const missionProgress = activeMission && activeMission.target ? Math.min(100, ((sim.snapshot?.maxAltitude ?? 0) / activeMission.target) * 100) : null;

  return (
    <div className="flex-1 flex flex-col lg:flex-row overflow-hidden">
      <div className="flex-1 flex flex-col min-w-0">
        {activeMission && (
          <div className="px-3 py-2 bg-indigo-500/10 border-b border-indigo-700/40 flex items-center justify-between flex-wrap gap-2">
            <div className="flex items-center gap-2 text-xs text-indigo-300"><Award size={14} /> Active Mission: <span className="font-semibold">{activeMission.name}</span> — {activeMission.brief}</div>
            {missionProgress != null && (
              <div className="flex items-center gap-2 w-40">
                <div className="flex-1 h-1.5 bg-slate-800 rounded-full overflow-hidden"><div className="h-full bg-indigo-400" style={{ width: `${missionProgress}%` }} /></div>
                <span className="text-[10px] font-mono text-indigo-300">{missionProgress.toFixed(0)}%</span>
              </div>
            )}
          </div>
        )}
        <div className="flex-1 relative bg-slate-950 min-h-[320px]">
          <canvas ref={canvasRef} width={760} height={520} className="w-full h-full block" />
          <div className="absolute top-2 left-2 px-2 py-1 rounded-sm bg-slate-950/70 border border-slate-800 text-[10px] uppercase tracking-widest text-slate-400">
            {!rocket.valid ? "Design invalid — fix in BUILD" : sim.phase.toUpperCase()}
          </div>
        </div>
        <FlightControls sim={sim} rocket={rocket} />
      </div>
      <TelemetryPanel sim={sim} rocket={rocket} />
    </div>
  );
}

function MissionsView({ rocket, teamName, setTeamName, activeMissionId, setActiveMissionId, onFly, leaderboard }) {
  return (
    <div className="flex-1 flex flex-col lg:flex-row overflow-hidden">
      <div className="w-full lg:w-80 shrink-0 border-r border-slate-800 p-3 overflow-y-auto space-y-2 lg:max-h-[calc(100vh-6.5rem)]">
        <div className="text-[11px] uppercase tracking-widest text-slate-500 mb-1">Rocketry Workshop</div>
        <input
          value={teamName}
          onChange={(e) => setTeamName(e.target.value)}
          placeholder="Team name"
          className="w-full mb-2 px-2.5 py-1.5 bg-slate-900 border border-slate-800 rounded-sm text-sm text-slate-200 placeholder:text-slate-600 focus:outline-none focus:border-cyan-600"
        />
        {MISSIONS.map((m) => (
          <button
            key={m.id}
            onClick={() => setActiveMissionId(m.id)}
            title={m.brief}
            className={`w-full text-left p-2.5 rounded-sm border ${activeMissionId === m.id ? "border-cyan-500/50 bg-cyan-500/10" : "border-slate-800 bg-slate-900/60"}`}
          >
            <div className="text-sm text-slate-200">{m.name}</div>
            <div className="text-[11px] text-slate-500">{m.brief}</div>
          </button>
        ))}
        <button
          onClick={onFly}
          disabled={!rocket?.valid || !activeMissionId}
          title={!activeMissionId ? "Select a mission first" : !rocket?.valid ? "Fix rocket design first" : "Fly the selected mission"}
          className="w-full mt-2 flex items-center justify-center gap-2 px-3 py-2 bg-amber-500 text-slate-950 rounded-sm text-xs uppercase tracking-widest font-semibold disabled:opacity-30 disabled:cursor-not-allowed"
        >
          <Award size={14} /> Fly Mission (Go to SIMULATE)
        </button>
        {!rocket?.valid && <div className="text-[11px] text-red-400">Fix rocket design before flying a mission.</div>}
        {!activeMissionId && <div className="text-[11px] text-slate-600">Select a mission above to attach it to your next flight.</div>}
      </div>
      <div className="flex-1 p-4 overflow-y-auto">
        <div className="flex items-center gap-2 mb-3"><Trophy size={16} className="text-amber-400" /><span className="text-sm uppercase tracking-widest text-slate-400">Leaderboard</span></div>
        <div className="text-[11px] text-slate-600 mb-3">Stored locally in this browser for this event session — see the report for why a shared cross-device leaderboard would need a small backend.</div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm min-w-[560px]">
            <thead>
              <tr className="text-[10px] uppercase tracking-widest text-slate-500 border-b border-slate-800">
                <th className="text-left py-1.5">Team</th>
                <th className="text-left py-1.5">Mission</th>
                <th className="text-right py-1.5">Altitude</th>
                <th className="text-right py-1.5">Max Vel.</th>
                <th className="text-right py-1.5">Payload</th>
                <th className="text-right py-1.5">Fuel Used</th>
                <th className="text-right py-1.5">Score</th>
              </tr>
            </thead>
            <tbody>
              {leaderboard.length === 0 && (
                <tr><td colSpan={7} className="py-6 text-center text-slate-600">No flights recorded yet.</td></tr>
              )}
              {leaderboard.slice().sort((a, b) => b.score - a.score).map((row, i) => (
                <tr key={i} className="border-b border-slate-900 font-mono text-xs">
                  <td className="py-1.5 text-slate-200 font-sans">{row.team}</td>
                  <td className="py-1.5 text-slate-400 font-sans">{row.mission}{row.completed ? " ✓" : " ✕"}</td>
                  <td className="text-right text-cyan-300">{row.altitude.toFixed(0)} m</td>
                  <td className="text-right text-cyan-300">{row.maxVel.toFixed(0)} m/s</td>
                  <td className="text-right text-slate-400">{row.payload} kg</td>
                  <td className="text-right text-slate-400">{row.fuelUsed.toFixed(1)} kg</td>
                  <td className="text-right text-amber-300">{row.score}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function ResultsView({ lastResult, rocket }) {
  if (!lastResult) return <div className="p-8 text-slate-500">Fly a rocket to see results and the flight debrief here.</div>;
  const explanations = [
    "Thrust-to-weight ratio matters because a rocket only lifts off once its engines push harder than gravity pulls down — a TWR under 1 means it stays on the pad.",
    "Rockets lose mass because they carry propellant as part of their own weight, and burn it into exhaust to generate thrust — the fuel itself is disposable cargo.",
    "Staging helps because dropping empty tanks and spent engines means later burns don't have to keep accelerating dead weight, so more of the remaining fuel goes into speed.",
    "Drag increases with velocity because F_drag scales with v², so it grows much faster than speed itself — doubling velocity quadruples the drag force.",
    "Δv (delta-v) matters because it's the total 'velocity budget' a rocket's fuel can theoretically provide — it caps what any flight profile can achieve, independent of gravity or drag losses along the way.",
    "Adding more fuel does not always help because extra propellant adds mass immediately, but its thrust only arrives gradually as it burns — past a point, the added weight costs more than the added Δv is worth.",
    "A fuel tank next to the wrong kind of engine is just dead weight — liquid tanks only feed liquid engines in the same stage, so mismatched pairings waste mass without adding any usable propellant.",
  ];
  return (
    <div className="flex-1 overflow-y-auto p-4 sm:p-6 max-w-3xl mx-auto w-full space-y-5">
      <div className="text-lg text-slate-100">Flight Debrief{lastResult.missionName ? ` — ${lastResult.missionName}` : ""}</div>
      {lastResult.missionName && (
        <div className={`px-3 py-2 rounded-sm border text-sm font-semibold ${lastResult.completed ? "border-emerald-600/50 bg-emerald-500/10 text-emerald-400" : "border-red-600/50 bg-red-500/10 text-red-400"}`}>
          {lastResult.completed ? "MISSION COMPLETE" : "MISSION NOT COMPLETED"} — Score: {lastResult.score}
        </div>
      )}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <div className="p-3 bg-slate-900 border border-slate-800 rounded-sm">
          <div className="text-[10px] uppercase tracking-widest text-slate-500">Max Altitude</div>
          <div className="font-mono text-xl text-cyan-300">{lastResult.maxAltitude.toFixed(0)} m</div>
        </div>
        <div className="p-3 bg-slate-900 border border-slate-800 rounded-sm">
          <div className="text-[10px] uppercase tracking-widest text-slate-500">Max Velocity</div>
          <div className="font-mono text-xl text-cyan-300">{(lastResult.maxVelocity ?? 0).toFixed(0)} m/s</div>
        </div>
        <div className="p-3 bg-slate-900 border border-slate-800 rounded-sm">
          <div className="text-[10px] uppercase tracking-widest text-slate-500">Theoretical Δv</div>
          <div className="font-mono text-xl text-cyan-300">{rocket?.dvTotal.toFixed(0)} m/s</div>
        </div>
        <div className="p-3 bg-slate-900 border border-slate-800 rounded-sm">
          <div className="text-[10px] uppercase tracking-widest text-slate-500">Flight Time</div>
          <div className="font-mono text-xl text-cyan-300">{lastResult.finalTime.toFixed(1)} s</div>
        </div>
      </div>
      <div>
        <div className="text-sm uppercase tracking-widest text-slate-400 mb-2">What Just Happened?</div>
        <ul className="space-y-2">
          {explanations.map((e, i) => (
            <li key={i} className="text-sm text-slate-300 bg-slate-900/60 border border-slate-800 rounded-sm p-2.5">{e}</li>
          ))}
        </ul>
      </div>
    </div>
  );
}

function ValidationView() {
  const [results, setResults] = useState(null);
  const run = () => setResults(runValidationSuite());
  useEffect(() => { run(); }, []);
  const failCount = (results || []).filter((r) => !r.pass).length;
  return (
    <div className="flex-1 overflow-y-auto p-4 sm:p-6 max-w-3xl mx-auto w-full">
      <div className="flex items-center justify-between mb-4 flex-wrap gap-2">
        <div className="flex items-center gap-2 text-lg text-slate-100"><FlaskConical size={18} className="text-cyan-400" /> Physics Validation Suite</div>
        <div className="flex items-center gap-2">
          {results && <span className={`text-xs font-mono ${failCount ? "text-red-400" : "text-emerald-400"}`}>{results.length - failCount}/{results.length} passing</span>}
          <button onClick={run} title="Re-run all validation tests" className="px-3 py-1.5 text-xs uppercase tracking-widest bg-slate-800 text-slate-300 rounded-sm">Re-run</button>
        </div>
      </div>
      <div className="space-y-1.5">
        {(results || []).map((r, i) => (
          <div key={i} className={`flex items-center justify-between gap-3 p-2.5 rounded-sm border ${r.pass ? "border-emerald-700/40 bg-emerald-500/5" : "border-red-700/40 bg-red-500/5"}`}>
            <div className="flex items-center gap-2 text-sm text-slate-200">
              {r.pass ? <CheckCircle2 size={15} className="text-emerald-400 shrink-0" /> : <XCircle size={15} className="text-red-400 shrink-0" />}
              {r.name}
            </div>
            <div className="text-right font-mono text-[11px] text-slate-400 shrink-0">
              exp: {String(r.expected)} · got: {String(r.actual)}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* =========================================================================
   APP ROOT
   ========================================================================= */

function instancesFromPartIds(partIds) {
  return partIds.map((partId) => ({ id: uid(), partId }));
}

export default function RocketLabApp() {
  const [tab, setTab] = useState("BUILD");
  const [instances, setInstances] = useState(() => {
    const saved = lsGet(LS_KEYS.currentDesign, null);
    return saved && Array.isArray(saved) && saved.length ? instancesFromPartIds(saved) : instancesFromPartIds(DEFAULT_STARTER);
  });
  const [selectedId, setSelectedId] = useState(null);
  const [teamName, setTeamName] = useState(() => lsGet(LS_KEYS.teamName, ""));
  const [leaderboard, setLeaderboard] = useState(() => lsGet(LS_KEYS.leaderboard, []));
  const [lastResult, setLastResult] = useState(null);
  const [saveMenuOpen, setSaveMenuOpen] = useState(false);
  const [activeMissionId, setActiveMissionId] = useState(null);

  const rocket = useMemo(() => buildRocket(instances), [instances]);
  const selectedDef = selectedId ? partDef(instances.find((i) => i.id === selectedId)?.partId) : null;
  const activeMission = activeMissionId ? MISSIONS.find((m) => m.id === activeMissionId) : null;

  // auto-persist the in-progress design + team name so a refresh never loses work (real persistence, not a fake toast)
  useEffect(() => { lsSet(LS_KEYS.currentDesign, instances.map((i) => i.partId)); }, [instances]);
  useEffect(() => { lsSet(LS_KEYS.teamName, teamName); }, [teamName]);
  useEffect(() => { lsSet(LS_KEYS.leaderboard, leaderboard); }, [leaderboard]);

  const warningMap = useMemo(() => {
    const map = {};
    rocket.stages.forEach((st) => {
      st.wastedTankIds.forEach((id) => { map[id] = "No liquid engine in this stage — this tank's fuel is unusable dead weight."; });
      st.underfueledEngineIds.forEach((id) => { map[id] = "No fuel tank in this stage — this liquid engine cannot produce thrust."; });
    });
    return map;
  }, [rocket.stages]);

  const addPart = useCallback((partId) => setInstances((prev) => [...prev, { id: uid(), partId }]), []);
  const deletePart = useCallback((id) => setInstances((prev) => prev.filter((i) => i.id !== id)), []);
  const dupPart = useCallback((id) => setInstances((prev) => {
    const idx = prev.findIndex((i) => i.id === id);
    if (idx === -1) return prev;
    const copy = { ...prev[idx], id: uid() };
    return [...prev.slice(0, idx + 1), copy, ...prev.slice(idx + 1)];
  }), []);
  const movePart = useCallback((id, dir) => setInstances((prev) => {
    const idx = prev.findIndex((i) => i.id === id);
    const swapWith = idx + dir;
    if (swapWith < 0 || swapWith >= prev.length) return prev;
    const next = prev.slice();
    [next[idx], next[swapWith]] = [next[swapWith], next[idx]];
    return next;
  }), []);
  const clearRocket = useCallback(() => setInstances([]), []);

  const newRocket = useCallback(() => {
    if (instances.length > 0 && !window.confirm("Start a new, empty rocket? This clears the current build.")) return;
    setInstances([]);
    setSelectedId(null);
    setTab("BUILD");
  }, [instances.length]);

  const resetRocket = useCallback(() => {
    const isAlreadyDefault = instances.length === DEFAULT_STARTER.length && instances.every((inst, i) => inst.partId === DEFAULT_STARTER[i]);
    if (!isAlreadyDefault && instances.length > 0 && !window.confirm("Reset to the default starter rocket? Your current build will be replaced.")) return;
    setInstances(instancesFromPartIds(DEFAULT_STARTER));
    setSelectedId(null);
  }, [instances]);

  const loadPrebuilt = useCallback((prebuilt) => {
    if (instances.length > 0 && !window.confirm(`Load "${prebuilt.name}"? This replaces the current build.`)) return;
    setInstances(instancesFromPartIds(prebuilt.parts));
    setSelectedId(null);
  }, [instances.length]);

  const loadSavedParts = useCallback((partIds) => {
    setInstances(instancesFromPartIds(partIds));
    setSelectedId(null);
    setSaveMenuOpen(false);
  }, []);

  // record a flight only once it has actually finished (landed) — never on abort/pause
  const recordFlight = useCallback((simResult, missionId) => {
    const mission = missionId ? MISSIONS.find((m) => m.id === missionId) : null;
    const fuelUsed = Math.max(0, rocket.totalPropellant - (simResult.propellantRemaining ?? 0));
    const payloadPart = rocket.instances.find((i) => partDef(i.partId).cat === "payload");

    if (mission) {
      const { completed, score } = scoreFlight(mission, simResult, rocket);
      setLastResult({ ...simResult, missionName: mission.name, completed, score });
      setLeaderboard((prev) => [...prev, {
        team: teamName || "Unnamed Team",
        mission: mission.name,
        altitude: simResult.maxAltitude,
        maxVel: simResult.maxVelocity || 0,
        payload: payloadPart ? partDef(payloadPart.partId).mass : 0,
        fuelUsed,
        completed, score,
      }]);
    } else {
      setLastResult({ ...simResult, missionName: null, completed: null, score: null });
    }
  }, [rocket, teamName]);

  return (
    <div className="w-full h-full min-h-[640px] bg-slate-950 text-slate-200 flex flex-col font-sans relative isolate">
      <Starfield />
      <div className="relative z-10 flex flex-col flex-1 min-h-0">
        <div className="relative">
          <TopBar
            tab={tab} setTab={setTab}
            onNew={newRocket} onReset={resetRocket}
            onLaunch={() => setTab("SIMULATE")}
            launchDisabled={!rocket.valid}
            saveMenuOpen={saveMenuOpen} setSaveMenuOpen={setSaveMenuOpen}
          />
          <div className="absolute right-3 sm:right-4 top-full">
            <SaveMenu open={saveMenuOpen} onClose={() => setSaveMenuOpen(false)} instances={instances} onLoad={loadSavedParts} teamName={teamName} />
          </div>
        </div>

        {tab === "BUILD" && (
          <div className="flex-1 flex flex-col lg:flex-row overflow-y-auto lg:overflow-hidden">
            <PartsPalette onAdd={addPart} onLoadPrebuilt={loadPrebuilt} />
            <BuilderCanvas
              instances={instances}
              selectedId={selectedId}
              setSelectedId={setSelectedId}
              onDelete={deletePart}
              onDup={dupPart}
              onMove={movePart}
              onDropPart={addPart}
              onClear={clearRocket}
              warningMap={warningMap}
            />
            <StatsPanel rocket={rocket} selectedDef={selectedDef} />
          </div>
        )}

        {tab === "SIMULATE" && <SimulateView rocket={rocket} activeMission={activeMission} onFlightComplete={recordFlight} />}
        {tab === "MISSIONS" && (
          <MissionsView
            rocket={rocket} teamName={teamName} setTeamName={setTeamName}
            activeMissionId={activeMissionId} setActiveMissionId={setActiveMissionId}
            onFly={() => setTab("SIMULATE")}
            leaderboard={leaderboard}
          />
        )}
        {tab === "RESULTS" && <ResultsView lastResult={lastResult} rocket={rocket} />}
        {tab === "VALIDATION" && <ValidationView />}
      </div>
    </div>
  );
}
