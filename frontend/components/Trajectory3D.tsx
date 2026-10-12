"use client";

import { Line, OrbitControls, Text } from "@react-three/drei";
import { Canvas } from "@react-three/fiber";
import { useMemo } from "react";

import { Empty } from "@/components/ui";
import type { Calibration, LbwAnalysis } from "@/lib/api";

// Pitch frame (metres): x across, y along the pitch towards the bowler, z up.
// three.js is y-up, so map (x, y, z) -> (x, z, -y).
const T = (p: number[]): [number, number, number] => [p[0], p[2], -p[1]];
const L = 20.12;
const HW = 0.1143;
const H = 0.7112;

function Stumps({ y }: { y: number }) {
  return (
    <group>
      {[-HW + 0.0175, 0, HW - 0.0175].map((x) => (
        <mesh key={x} position={T([x, y, H / 2])}>
          <cylinderGeometry args={[0.0175, 0.0175, H, 12]} />
          <meshStandardMaterial color="#e6e9ef" />
        </mesh>
      ))}
      <mesh position={T([0, y, H + 0.006])}>
        <boxGeometry args={[2 * HW, 0.012, 0.012]} />
        <meshStandardMaterial color="#fbbf24" />
      </mesh>
    </group>
  );
}

export default function Trajectory3D({ analysis, calibration }: { analysis: LbwAnalysis | null; calibration: Calibration | null }) {
  const r = analysis?.result;
  const observed = useMemo(() => (r?.observed_path_fit ?? []).map((p) => T(p.slice(1))), [r]);
  const predicted = useMemo(() => (r?.predicted_path ?? []).map((p) => T(p.slice(1))), [r]);
  const cam = useMemo(() => {
    if (!calibration) return null;
    const R = calibration.camera.R;
    const t = calibration.camera.t;
    // camera centre C = -R^T t
    const c = [0, 1, 2].map((i) => -(R[0][i] * t[0] + R[1][i] * t[1] + R[2][i] * t[2]));
    return c;
  }, [calibration]);

  if (!r) return <Empty>No 3D estimate yet. Run an LBW analysis with a camera calibration first. Uncalibrated video cannot be shown in 3D.</Empty>;
  const approximate = r.reconstruction_mode === "monocular_approximate" || !r.calibration_validated;
  const cross = r.wicket_projection?.crossing;

  return (
    <div className="space-y-2">
      {approximate && (
        <div className="rounded border border-warn/40 bg-warn/10 px-3 py-2 text-xs text-warn">
          APPROXIMATE 3D: single camera{!r.calibration_validated && ", unvalidated calibration"}. Depth is inferred from physical assumptions (gravity, drag, bounce, ball size), not
          measured by multiple views. Treat positions as estimates with the uncertainties shown in the LBW panel.
        </div>
      )}
      <div className="h-[560px] overflow-hidden rounded-lg border border-line bg-[#07090c]">
        <Canvas camera={{ position: [6, 4, 6], fov: 45 }}>
          <ambientLight intensity={0.6} />
          <directionalLight position={[5, 10, 5]} intensity={0.8} />
          <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.001, -L / 2]}>
            <planeGeometry args={[3.05, L + 4]} />
            <meshStandardMaterial color="#5b4a32" />
          </mesh>
          {[1.22, L - 1.22].map((y) => (
            <Line key={y} points={[T([-1.32, y, 0.002]), T([1.32, y, 0.002])]} color="#ffffff" lineWidth={1} />
          ))}
          <Stumps y={0} />
          <Stumps y={L} />
          {observed.length > 1 && <Line points={observed} color="#22d3ee" lineWidth={2.5} />}
          {predicted.length > 1 && <Line points={predicted} color="#e879f9" lineWidth={2.5} dashed dashSize={0.15} gapSize={0.1} />}
          {r.pitching_point && (
            <mesh position={T(r.pitching_point.mean)}>
              <sphereGeometry args={[0.05, 16, 16]} />
              <meshStandardMaterial color="#fbbf24" />
            </mesh>
          )}
          {r.impact_point && (
            <mesh position={T(r.impact_point.mean)}>
              <sphereGeometry args={[0.04, 16, 16]} />
              <meshStandardMaterial color="#f87171" />
            </mesh>
          )}
          {cross && (
            <mesh position={T([cross.mean[0], 0, cross.mean[1]])} scale={[Math.max(cross.std[0] * 2, 0.01), Math.max(cross.std[1] * 2, 0.01), 0.01]}>
              <sphereGeometry args={[1, 24, 24]} />
              <meshStandardMaterial color="#fb923c" transparent opacity={0.35} />
            </mesh>
          )}
          {cam && (
            <group position={T(cam)}>
              <mesh>
                <boxGeometry args={[0.4, 0.25, 0.25]} />
                <meshStandardMaterial color="#38bdf8" />
              </mesh>
              <Text position={[0, 0.4, 0]} fontSize={0.3} color="#38bdf8">
                {calibration?.camera.camera_id}
              </Text>
            </group>
          )}
          <OrbitControls target={T([0, 3, 0.5])} />
          <gridHelper args={[40, 40, "#1f2530", "#151a22"]} position={[0, -0.002, -L / 2]} />
        </Canvas>
      </div>
      <div className="text-xs text-muted">
        <span className="text-fitted">cyan</span>: fitted flight over observed span - <span className="text-predicted">magenta dashed</span>: predicted continuation after impact -{" "}
        <span className="text-interp">amber</span>: pitching point - <span className="text-bad">red</span>: impact - <span className="text-warn">orange</span>: 2-sigma region at the stumps -{" "}
        <span className="text-accent">blue box</span>: estimated camera. Drag to rotate, scroll to zoom.
      </div>
    </div>
  );
}
