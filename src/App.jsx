import React, { useState, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";
import katex from "katex";
import { ComposedChart, Line, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar, ErrorBar, ScatterChart, Scatter, ReferenceLine } from "recharts";

/* ─── Palette & Typography ───────────────────────────────────────────────── */
const K = {
  mb:  "var(--k-mb)", // modelblue
  gt:  "var(--k-gt)", // gradteal
  ab:  "var(--k-ab)", // avgblue
  np:  "var(--k-np)", // noisepink
  pp:  "var(--k-pp)", // privatepurp
  ag:  "var(--k-ag)", // arrowgray
  or:  "var(--k-or)", // orange
  gr:  "var(--k-gr)", // green
  bl:  "var(--k-bl)", // blue
  pu:  "var(--k-pu)", // purple
  rd:  "var(--k-rd)", // red
  bg:  "var(--k-bg)",
  pa:  "var(--k-pa)",
  bd:  "var(--k-bd)",
  ink: "var(--k-ink)",
  mu:  "var(--k-mu)",
  hi:  "var(--k-hi)",
};
const FZ = {
  tiny: "var(--fs-tiny)",
  xs: "var(--fs-xs)",
  sm: "var(--fs-sm)",
  md: "var(--fs-md)",
  lg: "var(--fs-lg)",
  xl: "var(--fs-xl)",
  xxl: "var(--fs-2xl)",
};
const SF = "Helvetica, sans-serif";

/* ═══════════════════════════════════════════════════════════════════════════ */
/* Diagrams                                                                    */
/* ═══════════════════════════════════════════════════════════════════════════ */
function DpSgdDiag() {
  return (
    <img src={`${import.meta.env.BASE_URL}dp_sgd.svg`} alt="Standard DP-SGD" style={{ display: "block", margin: "0 auto", width: "80%", height: "auto" }} />
  );
}


/* ═══════════════════════════════════════════════════════════════════════════ */
/* Diagram 2 – Naïve Rate Constraint                                          */
/* ═══════════════════════════════════════════════════════════════════════════ */
function NaiveDiag() {
  return (
    <img src={`${import.meta.env.BASE_URL}naive.svg`} alt="Naive Algorithm" style={{ display: "block", margin: "0 auto", width: "80%", height: "auto" }} />
  );
}


/* ═══════════════════════════════════════════════════════════════════════════ */
/* Diagram 3 – RaCO-DP                                                        */
/* Key TikZ insight: Loss is at the SAME x as Reg                            */
/*                   loss_grads is at the SAME x as reg_grads                */
/* ═══════════════════════════════════════════════════════════════════════════ */
function RacoDpDiag() {
  return (
    <img src={`${import.meta.env.BASE_URL}raco_dp.svg`} alt="RaCO-DP Algorithm" style={{ display: "block", margin: "0 auto", width: "100%", height: "auto" }} />
  );
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* App                                                                         */
/* ═══════════════════════════════════════════════════════════════════════════ */
const chartData = [
  { x: 0.040700, sgda: 0.041412, sgda_area: [0.038570, 0.044586] },
  { x: 0.042900, sgda: 0.036383, sgda_area: [0.032883, 0.039747] },
  { x: 0.043050, raco: 0.034250, raco_area: [0.025052, 0.042282] },
  { x: 0.043900, sgda: 0.028777, sgda_area: [0.021562, 0.034994] },
  { x: 0.044900, sgda: 0.024885, sgda_area: [0.020473, 0.029715] },
  { x: 0.045950, sgda: 0.019513, sgda_area: [0.012779, 0.025863] },
  { x: 0.046650, sgda: 0.015651, sgda_area: [0.009262, 0.020649], raco: 0.026576, raco_area: [0.019236, 0.035628] },
  { x: 0.047334, tran: 0.073333, tran_area: [0.069791, 0.076288], dpfermi: 0.073333, dpfermi_area: [0.065283, 0.081383] },
  { x: 0.047650, sgda: 0.019239, sgda_area: [0.013242, 0.025414] },
  { x: 0.048350, raco: 0.025275, raco_area: [0.020235, 0.029804] },
  { x: 0.049350, sgda: 0.015367, sgda_area: [0.011862, 0.018739] },
  { x: 0.049550, raco: 0.043725, raco_area: [0.005911, 0.069181] },
  { x: 0.049921, dpfermi: 0.065891, dpfermi_area: [0.056168, 0.075615] },
  { x: 0.050600, raco: 0.024459, raco_area: [0.016065, 0.033766] },
  { x: 0.051750, sgda: 0.009009, sgda_area: [0.006090, 0.012695] },
  { x: 0.052100, raco: 0.011156, raco_area: [0.005869, 0.017417] },
  { x: 0.054100, raco: 0.014014, raco_area: [0.006568, 0.022271] },
  { x: 0.055150, sgda: 0.002374, sgda_area: [0.000922, 0.004175] },
  { x: 0.056500, raco: 0.014659, raco_area: [0.009816, 0.019944] },
  { x: 0.057182, tran: 0.067790, tran_area: [0.063998, 0.071044] },
  { x: 0.058770, dpfermi: 0.047249, dpfermi_area: [0.039450, 0.055048] },
  { x: 0.059650, raco: 0.006434, raco_area: [0.002826, 0.010449] },
  { x: 0.065750, raco: 0.007780, raco_area: [0.004425, 0.011497] },
  { x: 0.086499, dpfermi: 0.018307, dpfermi_area: [0.011804, 0.024811] },
  { x: 0.117767, tran: 0.049481, tran_area: [0.045639, 0.052795] },
];

const gammaData01 = [{ x: 0.172049, y: 0.004216, xError: 0.001438, yError: 0.002621 }];
const gammaData05 = [{ x: 0.169133, y: 0.030170, xError: 0.008485, yError: 0.012582 }];
const gammaData10 = [{ x: 0.161723, y: 0.064874, xError: 0.007488, yError: 0.028123 }];
const gammaData15 = [{ x: 0.155139, y: 0.117723, xError: 0.004455, yError: 0.032007 }];
const gammaData20 = [{ x: 0.150666, y: 0.174611, xError: 0.000897, yError: 0.007985 }];

function GammaChart({ isRight }) {
  return (
    <div className={isRight ? "wrap-right" : ""} style={isRight ? { height: 350 } : { height: 400, width: "100%", margin: "32px 0" }}>
      <ResponsiveContainer>
        <ScatterChart margin={{ top: 20, right: 20, bottom: 30, left: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--k-bd)" vertical={false} />
          <XAxis 
            dataKey="x" 
            type="number" 
            domain={[0.145, 0.180]} 
            tick={{ fontSize: 12, fill: "var(--k-mu)" }} 
            tickFormatter={(val) => val.toFixed(3)}
            label={{ value: 'Test Error', position: 'insideBottom', offset: -15, fill: "var(--k-mu)", fontSize: 14 }} 
          />
          <YAxis 
            dataKey="y"
            type="number"
            domain={[0, 0.22]} 
            tick={{ fontSize: 12, fill: "var(--k-mu)" }}
            tickFormatter={(val) => val.toFixed(2)}
            label={{ value: 'Constraint Value', angle: -90, position: 'insideLeft', offset: -2, dy: 50, fill: "var(--k-mu)", fontSize: 14 }} 
          />
          <Tooltip 
            cursor={{ strokeDasharray: '3 3' }}
            contentStyle={{ backgroundColor: "var(--k-pa)", borderColor: "var(--k-bd)", borderRadius: 6, color: "var(--k-ink)", fontSize: 13 }}
            formatter={(value, name) => [Number(value).toFixed(4), name]}
            labelFormatter={(label) => `Test Error: ${Number(label).toFixed(4)}`}
          />
          <Legend wrapperStyle={{ fontSize: 13, paddingTop: 20 }} />

          <ReferenceLine y={0.01} stroke={K.bl} strokeDasharray="5 5" opacity={0.5} />
          <Scatter name="γ = 0.01" data={gammaData01} fill={K.bl} shape="circle">
            <ErrorBar dataKey="xError" width={4} strokeWidth={2} stroke={K.bl} opacity={0.8} direction="x" />
            <ErrorBar dataKey="yError" width={4} strokeWidth={2} stroke={K.bl} opacity={0.8} direction="y" />
          </Scatter>

          <ReferenceLine y={0.05} stroke={K.or} strokeDasharray="5 5" opacity={0.5} />
          <Scatter name="γ = 0.05" data={gammaData05} fill={K.or} shape="square">
            <ErrorBar dataKey="xError" width={4} strokeWidth={2} stroke={K.or} opacity={0.8} direction="x" />
            <ErrorBar dataKey="yError" width={4} strokeWidth={2} stroke={K.or} opacity={0.8} direction="y" />
          </Scatter>

          <ReferenceLine y={0.10} stroke={K.gt} strokeDasharray="5 5" opacity={0.5} />
          <Scatter name="γ = 0.10" data={gammaData10} fill={K.gt} shape="triangle">
            <ErrorBar dataKey="xError" width={4} strokeWidth={2} stroke={K.gt} opacity={0.8} direction="x" />
            <ErrorBar dataKey="yError" width={4} strokeWidth={2} stroke={K.gt} opacity={0.8} direction="y" />
          </Scatter>

          <ReferenceLine y={0.15} stroke={K.pp} strokeDasharray="5 5" opacity={0.5} />
          <Scatter name="γ = 0.15" data={gammaData15} fill={K.pp} shape="diamond">
            <ErrorBar dataKey="xError" width={4} strokeWidth={2} stroke={K.pp} opacity={0.8} direction="x" />
            <ErrorBar dataKey="yError" width={4} strokeWidth={2} stroke={K.pp} opacity={0.8} direction="y" />
          </Scatter>

          <ReferenceLine y={0.20} stroke={K.np} strokeDasharray="5 5" opacity={0.5} />
          <Scatter name="γ = 0.20" data={gammaData20} fill={K.np} shape="cross">
            <ErrorBar dataKey="xError" width={4} strokeWidth={2} stroke={K.np} opacity={0.8} direction="x" />
            <ErrorBar dataKey="yError" width={4} strokeWidth={2} stroke={K.np} opacity={0.8} direction="y" />
          </Scatter>
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}

const barChartData = [
  { name: "Unconstrained", sgda: 0.9681, sgda_err: 0.0007, raco9: 0.9171, raco9_err: 0.0020, raco1: 0.9006, raco1_err: 0.0077 },
  { name: "0.1", sgda: 0.9580, sgda_err: 0.0130, raco9: 0.9168, raco9_err: 0.0170, raco1: 0.9000, raco1_err: 0.0250 },
  { name: "0.075", sgda: 0.9500, sgda_err: 0.0007, raco9: 0.9128, raco9_err: 0.0017, raco1: 0.8940, raco1_err: 0.0046 },
  { name: "0.05", sgda: 0.9380, sgda_err: 0.0009, raco9: 0.9090, raco9_err: 0.0034, raco1: 0.8877, raco1_err: 0.0065 },
];

function ResultsChart() {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(400px, 1fr))", gap: 32, marginTop: 32, marginBottom: 24 }}>
      <figure style={{ margin: 0, width: "100%" }}>
        <div style={{ height: 400, width: "100%" }}>
          <ResponsiveContainer>
          <ComposedChart data={chartData} margin={{ top: 20, right: 20, bottom: 30, left: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--k-bd)" vertical={false} />
            <XAxis 
              dataKey="x" 
              type="number" 
              domain={[0.036, 0.122]} 
              tick={{ fontSize: 12, fill: "var(--k-mu)" }} 
              tickFormatter={(val) => val.toFixed(2)}
              label={{ value: 'Test Error', position: 'insideBottom', offset: -15, fill: "var(--k-mu)", fontSize: 14 }} 
            />
            <YAxis 
              domain={[-0.003, 0.086]} 
              tick={{ fontSize: 12, fill: "var(--k-mu)" }}
              tickFormatter={(val) => val.toFixed(2)}
              // label={{ value: 'Demographic Disparity', angle: -90, position: 'insideLeft', offset: -10, fill: "var(--k-mu)", fontSize: 14 }} 
            label={{ value: 'Demographic Disparity', angle: -90, position: 'insideLeft', offset: -2, dy: 60, fill: "var(--k-mu)", fontSize: 14 }} 
            />
            <Tooltip 
              labelFormatter={(label) => `Test Error: ${Number(label).toFixed(4)}`}
              contentStyle={{ backgroundColor: "var(--k-pa)", borderColor: "var(--k-bd)", borderRadius: 6, color: "var(--k-ink)", fontSize: 13 }}
            />
            <Legend wrapperStyle={{ fontSize: 13, paddingTop: 20 }} />

            {/* We hide the Area from tooltips/legends so it doesn't double-up with the lines */}
            <Area type="monotone" dataKey="sgda_area" fill={K.ag} stroke="none" fillOpacity={0.2} connectNulls tooltipType="none" legendType="none" />
            <Line type="monotone" dataKey="sgda" name="SGDA (Non-Private)" stroke={K.ag} strokeWidth={2} dot={{ r: 4, fill: K.ag }} activeDot={{ r: 6 }} connectNulls />

            <Area type="monotone" dataKey="tran_area" fill={K.or} stroke="none" fillOpacity={0.3} connectNulls tooltipType="none" legendType="none" />
            <Line type="monotone" dataKey="tran" name="Tran et al. (2021)" stroke={K.or} strokeWidth={2} dot={{ r: 4, fill: K.or }} activeDot={{ r: 6 }} connectNulls />

            <Area type="monotone" dataKey="dpfermi_area" fill={K.bl} stroke="none" fillOpacity={0.3} connectNulls tooltipType="none" legendType="none" />
            <Line type="monotone" dataKey="dpfermi" name="DP-FERMI (2023)" stroke={K.bl} strokeWidth={2} dot={{ r: 4, fill: K.bl }} activeDot={{ r: 6 }} connectNulls />

            <Area type="monotone" dataKey="raco_area" fill={K.pp} stroke="none" fillOpacity={0.3} connectNulls tooltipType="none" legendType="none" />
            <Line type="monotone" dataKey="raco" name="RaCO-DP" stroke={K.pp} strokeWidth={2} dot={{ r: 4, fill: K.pp }} activeDot={{ r: 6 }} connectNulls />
          </ComposedChart>
          </ResponsiveContainer>
        </div>
        <figcaption style={{ textAlign: "center", color: K.mu, fontSize: FZ.sm, marginTop: 12 }}>
          Parkinsons dataset.
          Logistic Regressions models trained with <TeX>{"\\varepsilon=1"}</TeX>
        </figcaption>
      </figure>

      <figure style={{ margin: 0, width: "100%" }}>
        <div style={{ height: 400, width: "100%" }}>
          <ResponsiveContainer>
          <BarChart data={barChartData} margin={{ top: 20, right: 20, bottom: 30, left: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--k-bd)" vertical={false} />
            <XAxis 
              dataKey="name" 
              tick={{ fontSize: 12, fill: "var(--k-mu)" }} 
              label={{ value: 'Demographic Parity Constraint (γ)', position: 'insideBottom', offset: -15, fill: "var(--k-mu)", fontSize: 14 }} 
            />
            <YAxis 
              domain={[0.8, 1.01]} 
              tick={{ fontSize: 12, fill: "var(--k-mu)" }}
              tickFormatter={(val) => val.toFixed(2)}
              label={{ value: 'Accuracy', angle: -90, position: 'insideLeft', offset: -5, fill: "var(--k-mu)", fontSize: 14 }} 
            />
            <Tooltip 
              cursor={{ fill: "var(--k-bd)", opacity: 0.4 }}
              contentStyle={{ backgroundColor: "var(--k-pa)", borderColor: "var(--k-bd)", borderRadius: 6, color: "var(--k-ink)", fontSize: 13 }}
              formatter={(value, name) => [value.toFixed(4), name]}
            />
            <Legend wrapperStyle={{ fontSize: 13, paddingTop: 20 }} />

            <Bar dataKey="sgda" name="SGDA (Non-Private)" fill={K.ag} radius={[2,2,0,0]}>
              <ErrorBar dataKey="sgda_err" width={4} strokeWidth={2} stroke="var(--k-ink)" opacity={0.6} />
            </Bar>
            <Bar dataKey="raco9" name="RaCO-DP (ε=9)" fill={K.pp} fillOpacity={0.6} radius={[2,2,0,0]}>
              <ErrorBar dataKey="raco9_err" width={4} strokeWidth={2} stroke="var(--k-ink)" opacity={0.6} />
            </Bar>
            <Bar dataKey="raco1" name="RaCO-DP (ε=1)" fill={K.pp} radius={[2,2,0,0]}>
              <ErrorBar dataKey="raco1_err" width={4} strokeWidth={2} stroke="var(--k-ink)" opacity={0.6} />
            </Bar>
          </BarChart>
          </ResponsiveContainer>
        </div>
        <figcaption style={{ textAlign: "center", color: K.mu, fontSize: FZ.sm, marginTop: 12 }}>
          CelebA using a ResNet-16 model
        </figcaption>
      </figure>
    </div>
  );
}

const steps = [
  {
    id: "dp-sgd",
    title: "Standard DP-SGD",
    desc: "Training decomposes cleanly over examples. Each sample contributes exactly 1 gradient term, so we can clip per-sample gradients to bound sensitivity, average them, and add a calibrated Gaussian noise. The model loop closes with a parameter update.",
    diag: <DpSgdDiag />,
    nextLabel: "Can't we add a regularizer? →",
  },
  {
    id: "naive",
    title: "Naïve rate constraint",
    desc: <>Adding a regularizer <TeX>{"R(\\theta, D)"}</TeX> to enforce a rate constraint looks natural — but the regularizer depends on the entire dataset <TeX>D</TeX>. This breaks per-sample decomposability: each sample can contribute up to <TeX>{"|D| + 1"}</TeX> terms, making sensitivity <TeX>{"\\mathcal{O}(|D|)"}</TeX> and requiring far too much noise.</>,
    diag: <NaiveDiag />,
    prevLabel: "← Back to Standard DP-SGD",
    nextLabel: "Remove the direct dependency on the dataset →",
  },
  {
    id: "raco",
    title: "RaCO-DP",
    desc: "Rate constraints only need prediction rates over subgroups, i.e., a histogram. RaCO-DP privatizes the histogram once per step with Laplace noise (sensitivity = 1). All gradient and constraint computations then flow from this private histogram via post-processing, incurring zero additional privacy cost.",
    diag: <RacoDpDiag />,
    prevLabel: "← Without a histogram",
  },
];

const Box = ({ children, style }) => (
  <div style={{ border: `1px solid ${K.bd}`, borderRadius: 6, background: K.pa, ...style }}>
    {children}
  </div>
);
const Lbl = ({ children, color = K.mu }) => (
  <span style={{ fontSize: FZ.lg, letterSpacing: "0.06em", textTransform: "uppercase", color }}>
    {children}
  </span>
);

function TeX({ children, block = false }) {
  const html = katex.renderToString(String(children), {
    throwOnError: false,
    displayMode: block,
  });
  return <span dangerouslySetInnerHTML={{ __html: html }} />;
}

function CitationBlock() {
  const [copied, setCopied] = useState(false);
  const bibtex = `@inproceedings{
  yaghini2026private,
  title={Private Rate-Constrained Optimization with Applications to Fair Learning},
  author={Mohammad Yaghini and Tudor Cebere and Michael Menart and Aur{\\'e}lien Bellet and Nicolas Papernot},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=mex3rvs2KX}
}`;

  const handleCopy = () => {
    navigator.clipboard.writeText(bibtex);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div style={{ marginTop: 64, borderTop: `1px solid ${K.bd}`, paddingTop: 40 }}>
      <Lbl>Cite us</Lbl>
      <div style={{ position: "relative", marginTop: 16 }}>
        <pre style={{ margin: 0, padding: 20, background: K.pa, borderRadius: 6, border: `1px solid ${K.bd}`, overflowX: "auto", fontSize: FZ.sm, lineHeight: 1.5, color: K.ink }}>
          <code>{bibtex}</code>
        </pre>
        <button onClick={handleCopy} style={{
          position: "absolute", top: 12, right: 12, padding: "6px 12px",
          background: K.bg, border: `1px solid ${K.bd}`, borderRadius: 4,
          cursor: "pointer", fontSize: FZ.xs, fontWeight: 500,
          color: copied ? K.gr : K.mu, transition: "all 0.2s ease"
        }}>
          {copied ? "Copied!" : "Copy"}
        </button>
      </div>
    </div>
  );
}

export default function App() {
  const [tab, setTab] = useState("walkthrough");
  const [activeStep, setActiveStep] = useState(0);
  const [blogContent, setBlogContent] = useState("");

  useEffect(() => {
    if (tab === "blog" && !blogContent) {
      fetch(`${import.meta.env.BASE_URL}blog.md`)
        .then((res) => res.text())
        .then((text) => setBlogContent(text))
        .catch(() => setBlogContent("Failed to load blog post."));
    }
  }, [tab, blogContent]);

  return (
    <div style={{ fontFamily: SF, background: K.bg, minHeight: "100vh", color: K.ink, textAlign: "left" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,300;8..60,400;8..60,600&display=swap');
        body { margin: 0; } * { box-sizing: border-box; } button { cursor: pointer; font-family: inherit; }
      `}</style>

      {/* Header */}
      <div style={{
        borderBottom: `1px solid ${K.bd}`, padding: "28px 40px",
        background: K.pa, display: "flex", justifyContent: "space-between",
        alignItems: "baseline", flexWrap: "wrap", gap: 12,
        
      }}>
        <div>
          <div style={{ fontSize: 30, fontWeight: 600, letterSpacing: "-0.01em" }}>
            Private Rate-Constrained Optimization
          </div>
          <div style={{ fontSize: FZ.md, color: K.mu, marginTop: 4 }}>
            Yaghini* · Cebere* · Menart · Bellet · Papernot &nbsp;·&nbsp; ICLR 2026
          </div>

          {/* Tabs */}
          <div style={{ display: "flex", gap: 16, marginTop: 20 }}>
            <button onClick={() => setTab("walkthrough")} style={{
              background: "none", border: "none", padding: 0,
              fontSize: FZ.md, fontWeight: tab === "walkthrough" ? 600 : 400,
              color: tab === "walkthrough" ? K.bl : K.mu,
              textDecoration: tab === "walkthrough" ? "underline" : "none",
              textUnderlineOffset: 4
            }}>
              Walkthrough
            </button>
            <button onClick={() => setTab("blog")} style={{
              background: "none", border: "none", padding: 0,
              fontSize: FZ.md, fontWeight: tab === "blog" ? 600 : 400,
              color: tab === "blog" ? K.bl : K.mu,
              textDecoration: tab === "blog" ? "underline" : "none",
              textUnderlineOffset: 4
            }}>
              Blog
            </button>
          </div>
        </div>
        <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
          <a href="https://openreview.net/forum?id=mex3rvs2KX" target="_blank" rel="noopener noreferrer"
            style={{ padding: "8px 16px", borderRadius: 6, background: K.ink, color: K.pa, textDecoration: "none", fontSize: FZ.sm, fontWeight: 500, transition: "all 0.2s ease" }}>
            Paper ↗
          </a>
          <a href="https://iclr.cc/virtual/2026/poster/10007554" target="_blank" rel="noopener noreferrer"
            style={{ padding: "8px 16px", borderRadius: 6, border: `1px solid ${K.bd}`, background: K.pa, color: K.ink, textDecoration: "none", fontSize: FZ.sm, fontWeight: 500, transition: "all 0.2s ease" }}>
            ICLR Page ↗
          </a>
          <a href="https://github.com/cleverhans-lab/dp-raco" target="_blank" rel="noopener noreferrer"
            style={{ padding: "8px 16px", borderRadius: 6, border: `1px solid ${K.bd}`, background: K.pa, color: K.ink, textDecoration: "none", fontSize: FZ.sm, fontWeight: 500, transition: "all 0.2s ease" }}>
            GitHub ↗
          </a>
        </div>
      </div>

      {/* Main */}
      <div style={{ maxWidth: 980, margin: "0 auto", padding: "48px 24px" }}>

        {tab === "walkthrough" && (
          <>
            <p style={{ fontSize: FZ.lg, lineHeight: 1.8, color: K.ink, margin: "0 0 48px" }}>
              Many ML requirements, such as fairness or robustness constraints, 
              can be written as <em>rate constraints</em>. Training under these with differential privacy
              is hard because they break the per-sample structure DP-SGD relies on.
              <strong> RaCO-DP</strong> solves this by privatizing a histogram of model predictions.
            </p>

            <div style={{ marginTop: 24 }}>
              {/* Segmented Control */}
              <div style={{
                display: "flex", gap: 4, marginBottom: 32, background: "var(--k-bd)",
                padding: 4, borderRadius: 8, width: "fit-content"
              }}>
                {steps.map((s, i) => (
                  <button
                    key={s.id}
                    onClick={() => setActiveStep(i)}
                    style={{
                      padding: "8px 16px", borderRadius: 6, border: "none",
                      background: activeStep === i ? "var(--k-pa)" : "transparent",
                      color: activeStep === i ? "var(--k-ink)" : "var(--k-mu)",
                      fontWeight: activeStep === i ? 600 : 500, fontSize: FZ.sm,
                      boxShadow: activeStep === i ? "0 1px 3px rgba(0,0,0,0.1)" : "none",
                      transition: "all 0.2s ease"
                    }}
                  >
                    {s.title}
                  </button>
                ))}
              </div>

              {/* Active Step Content */}
              <div style={{ position: "relative" }}>
                {steps.map((s, i) => {
                  const isActive = activeStep === i;
                  return (
                    <div
                      key={s.id}
                      style={{
                        opacity: isActive ? 1 : 0,
                        pointerEvents: isActive ? "auto" : "none",
                        position: isActive ? "relative" : "absolute",
                        top: 0, left: 0, width: "100%",
                        transition: "opacity 0.4s ease",
                        zIndex: isActive ? 1 : 0,
                      }}
                    >
                      <div style={{ marginBottom: 16 }}>
                        <div style={{ fontWeight: 600, fontSize: FZ.xl, marginBottom: 6 }}>{s.title}</div>
                        <div style={{ fontSize: FZ.md, color: K.mu, lineHeight: 1.6 }}>{s.desc}</div>
                      </div>

                      <Box style={{ overflowX: "auto", padding: "28px 20px 20px" }}>
                        {s.diag}
                      </Box>

                      {s.id === "naive" && (
                        <div style={{
                          marginTop: 12, padding: "12px 16px",
                          background: "var(--k-err-bg)", border: `1px solid var(--k-err-bd)`, borderRadius: 6,
                        }}>
                          <div style={{ fontWeight: 600, color: K.rd, fontSize: FZ.sm, marginBottom: 4 }}>
                            Problem: <TeX>{"R(\\theta; D)"}</TeX> depends on the whole dataset
                          </div>
                          <div style={{ fontSize: FZ.xs, color: K.ink, lineHeight: 1.65 }}>
                            The per-sample gradient of sample <em>i</em> receives contributions from every other
                            sample <em>j</em> through the shared constraint. Each sample contributes up to{" "}
                            <strong><TeX>{"|D| + 1"}</TeX> terms</strong> — requiring noise proportional to dataset size and
                            destroying utility.
                          </div>
                        </div>
                      )}

                      {/* Prev / Next controls */}
                      <div style={{ display: "flex", justifyContent: "space-between", marginTop: 24 }}>
                        {s.prevLabel ? (
                          <button onClick={() => setActiveStep(i - 1)} style={{
                            padding: "8px 20px", borderRadius: 6, border: `1px solid var(--k-bd)`,
                            background: "var(--k-pa)", color: "var(--k-ink)",
                            cursor: "pointer", fontSize: FZ.sm, fontWeight: 500, transition: "all 0.2s ease"
                          }}>
                            {s.prevLabel}
                          </button>
                        ) : <div />}

                        {s.nextLabel ? (
                          <button onClick={() => setActiveStep(i + 1)} style={{
                            padding: "8px 20px", borderRadius: 6, border: `1px solid var(--k-ink)`,
                            background: "var(--k-ink)", color: "var(--k-pa)",
                            cursor: "pointer", fontSize: FZ.sm, fontWeight: 500, transition: "all 0.2s ease"
                          }}>
                            {s.nextLabel}
                          </button>
                        ) : <div />}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </>
        )}

        {tab === "blog" && (
          <div className="blog-content" style={{ fontSize: FZ.lg, lineHeight: 1.8, color: K.ink }}>
            <ReactMarkdown
              remarkPlugins={[remarkMath]}
              rehypePlugins={[rehypeKatex]}
              components={{
                img: ({ node, ...props }) => {
                  if (props.alt && props.alt.includes("Gamma Chart")) return <GammaChart isRight={props.alt.includes("right")} />;
                  return <img {...props} />;
                }
              }}
            >
              {blogContent}
            </ReactMarkdown>
          </div>
        )}

        {/* Key results */}
        <div style={{ marginTop: 64, borderTop: `1px solid ${K.bd}`, paddingTop: 40 }}>
          <Lbl>Main Results</Lbl>
            <br/>
            <br/>
            <p style={{ fontSize: FZ.lg, lineHeight: 1.8, color: K.ink, margin: "0 0 48px" }}>
            On tabular data, RaCO-DP Pareto dominates prior SOTA and
            nearly closes the optimality gap with non-private models.
            On deep models, our method maintains high utility even at small <TeX>{"\\varepsilon"}</TeX>
while reliably satisfying fairness constraints
            </p>
          <ResultsChart />

          {/* <div style={{
            marginTop: 32, display: "grid",
            gridTemplateColumns: "repeat(auto-fill, minmax(190px, 1fr))",
            border: `1px solid ${K.bd}`, borderRadius: 6, overflow: "hidden",
          }}>
            {[
              { val: "≈ 0",     label: "extra privacy cost over DP-SGD" },
              { val: "Q + 1",   label: "terms per sample (Q = # subgroups)" },
              { val: "1000×",   label: "faster than DP-FERMI" },
              { val: "direct γ",label: "constraint satisfaction, no hyperparameter tuning" },
            ].map((s, i, arr) => (
              <div key={s.label} style={{
                background: K.pa, padding: "24px 20px",
                borderRight: i < arr.length - 1 ? `1px solid ${K.bd}` : "none",
              }}>
                <div style={{ fontSize: 28, fontWeight: 600, fontFamily: "'Source Serif 4', Georgia, serif", letterSpacing: "-0.02em" }}>
                  {s.val}
                </div>
                <div style={{ fontSize: FZ.xs, color: K.mu, marginTop: 6, lineHeight: 1.5 }}>{s.label}</div>
              </div>
            ))}
          </div> */}

          <CitationBlock />
        </div>
      </div>

      {/* Footer */}
      <div style={{
        borderTop: `1px solid ${K.bd}`, padding: "20px 40px",
        background: K.pa, fontSize: FZ.xs, color: K.mu,
        display: "flex", justifyContent: "space-between", flexWrap: "wrap", gap: 8,
      }}>
        <span>ICLR 2026 · University of Toronto · Vector Institute · Inria</span>
        <div style={{ display: "flex", gap: 16 }}>
          <a href="https://openreview.net/forum?id=mex3rvs2KX" target="_blank" rel="noopener noreferrer" style={{ color: K.bl, textDecoration: "none" }}>Paper ↗</a>
          <a href="https://iclr.cc/virtual/2026/poster/10007554" target="_blank" rel="noopener noreferrer" style={{ color: K.bl, textDecoration: "none" }}>ICLR Page ↗</a>
          <a href="https://github.com/cleverhans-lab/dp-raco" target="_blank" rel="noopener noreferrer" style={{ color: K.bl, textDecoration: "none" }}>Code ↗</a>
        </div>
      </div>
    </div>
  );
}
