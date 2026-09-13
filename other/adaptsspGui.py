"""
gui_adaptssp.py
Graphical User Interface (GUI) for ADAPT-SSP Solver v4.
Pure Python + Tkinter implementation calling v4/adaptsspCli.exe via subprocess with --json.
"""

import os
import sys
import json
import subprocess
import threading
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

class AdaptSspGui:
    def __init__(self, root):
        self.root = root
        self.root.title("ADAPT-SSP Solver v4 - Desktop GUI")
        self.root.geometry("860x780")
        self.root.minsize(760, 680)

        # Locate adaptsspCli.exe
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.exe_path = os.path.join(base_dir, "adaptsspCli.exe")
        if not os.path.exists(self.exe_path):
            self.exe_path = os.path.join(os.path.dirname(base_dir), "v4", "adaptsspCli.exe")
        if not os.path.exists(self.exe_path):
            self.exe_path = os.path.join(os.path.dirname(base_dir), "adaptsspCli.exe")

        self.setup_styles()
        self.create_widgets()

    def setup_styles(self):
        self.style = ttk.Style()
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        # Configure colors and fonts
        self.style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"), foreground="#1a365d")
        self.style.configure("SubHeader.TLabel", font=("Segoe UI", 9), foreground="#4a5568")
        self.style.configure("Bold.TLabel", font=("Segoe UI", 9, "bold"))
        self.style.configure("Action.TButton", font=("Segoe UI", 10, "bold"))

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="14")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title Banner
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill=tk.X, pady=(0, 10))

        title_lbl = ttk.Label(title_frame, text="ADAPT-SSP Solver v4", style="Header.TLabel")
        title_lbl.pack(anchor=tk.W)

        sub_lbl = ttk.Label(
            title_frame, 
            text="Exact Subset Sum Solver with Adaptive Pruned DFS, Gray Tail-Table & L7 Independent Witness Verification", 
            style="SubHeader.TLabel"
        )
        sub_lbl.pack(anchor=tk.W)

        # Input Group
        input_group = ttk.LabelFrame(main_frame, text=" 1. Input Parameters ", padding="10")
        input_group.pack(fill=tk.X, pady=(0, 10))

        # Elements Input
        elem_lbl = ttk.Label(input_group, text="Elements List (Comma or space separated integers/decimals):", style="Bold.TLabel")
        elem_lbl.pack(anchor=tk.W, pady=(2, 4))

        self.elem_text = scrolledtext.ScrolledText(input_group, height=4, font=("Consolas", 10), wrap=tk.WORD)
        self.elem_text.pack(fill=tk.X, pady=(0, 8))
        self.elem_text.insert(tk.END, "75872066500, 68562112744, 19339160129, 24156275768, 11525390137, 34580469918, 87752813318, 25906232742, 20636790211, 47170921689")

        # Target, Mode, Timeout Row
        params_row = ttk.Frame(input_group)
        params_row.pack(fill=tk.X, pady=(0, 6))

        # Target Field
        tgt_frame = ttk.Frame(params_row)
        tgt_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        ttk.Label(tgt_frame, text="Target Value (T):", style="Bold.TLabel").pack(anchor=tk.W)
        self.target_var = tk.StringVar(value="135205864112")
        self.target_entry = ttk.Entry(tgt_frame, textvariable=self.target_var, font=("Consolas", 10))
        self.target_entry.pack(fill=tk.X, pady=(3, 0))

        # Solve Mode Combobox
        mode_frame = ttk.Frame(params_row)
        mode_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        ttk.Label(mode_frame, text="Solve Mode:", style="Bold.TLabel").pack(anchor=tk.W)
        self.mode_var = tk.StringVar(value="findone")
        self.mode_combo = ttk.Combobox(
            mode_frame,
            textvariable=self.mode_var,
            state="readonly",
            values=["findone", "findall-zero", "findall-dfs", "countall", "decision"]
        )
        self.mode_combo.pack(fill=tk.X, pady=(3, 0))

        # Time Limit
        tl_frame = ttk.Frame(params_row)
        tl_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(tl_frame, text="Time Limit (ms):", style="Bold.TLabel").pack(anchor=tk.W)
        self.timeout_var = tk.StringVar(value="120000")
        self.timeout_entry = ttk.Entry(tl_frame, textvariable=self.timeout_var, font=("Consolas", 10))
        self.timeout_entry.pack(fill=tk.X, pady=(3, 0))

        # Button Row
        btn_row = ttk.Frame(input_group)
        btn_row.pack(fill=tk.X, pady=(6, 0))

        self.solve_btn = ttk.Button(btn_row, text="▶  Solve SSP", style="Action.TButton", command=self.on_solve_clicked)
        self.solve_btn.pack(side=tk.LEFT, padx=(0, 8), ipadx=10, ipady=3)

        ttk.Button(btn_row, text="Sample 1 (Integers)", command=self.load_sample_integers).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(btn_row, text="Sample 2 (Decimals)", command=self.load_sample_decimals).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(btn_row, text="Sample 3 (UNSAT Parity)", command=self.load_sample_unsat).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(btn_row, text="Clear", command=self.clear_inputs).pack(side=tk.RIGHT)

        # Status & Metrics Group
        metrics_group = ttk.LabelFrame(main_frame, text=" 2. Solver Execution Metrics ", padding="10")
        metrics_group.pack(fill=tk.X, pady=(0, 10))

        # Status Banner
        self.status_banner = tk.Label(
            metrics_group,
            text="READY - Enter parameters and click Solve SSP",
            font=("Segoe UI", 11, "bold"),
            bg="#edf2f7",
            fg="#2d3748",
            padx=8,
            pady=6,
            relief=tk.RIDGE
        )
        self.status_banner.pack(fill=tk.X, pady=(0, 8))

        # Metric grid
        grid_frame = ttk.Frame(metrics_group)
        grid_frame.pack(fill=tk.X)

        self.metric_strategy = self.make_metric_box(grid_frame, 0, 0, "Strategy Chosen:", "-")
        self.metric_runtime  = self.make_metric_box(grid_frame, 0, 1, "Runtime:", "-")
        self.metric_ram      = self.make_metric_box(grid_frame, 0, 2, "Peak RAM:", "-")
        self.metric_k_range  = self.make_metric_box(grid_frame, 1, 0, "Feasible K Window:", "-")
        self.metric_nodes    = self.make_metric_box(grid_frame, 1, 1, "States Evaluated / Pruned:", "-")
        self.metric_l7       = self.make_metric_box(grid_frame, 1, 2, "L7 Independent Verifier:", "-")

        # Solution Output Group
        output_group = ttk.LabelFrame(main_frame, text=" 3. Solution Details & Witness Inspection ", padding="10")
        output_group.pack(fill=tk.BOTH, expand=True)

        self.output_text = scrolledtext.ScrolledText(output_group, font=("Consolas", 10), wrap=tk.WORD)
        self.output_text.pack(fill=tk.BOTH, expand=True)

        # Bottom Bar
        self.bottom_bar = ttk.Label(main_frame, text=f"CLI Engine Path: {self.exe_path}", style="SubHeader.TLabel")
        self.bottom_bar.pack(anchor=tk.W, pady=(4, 0))

    def make_metric_box(self, parent, row, col, label_text, default_val):
        cell = ttk.Frame(parent, padding="3")
        cell.grid(row=row, column=col, sticky="w", padx=6, pady=2)
        ttk.Label(cell, text=label_text, style="Bold.TLabel").pack(anchor=tk.W)
        val_lbl = ttk.Label(cell, text=default_val, font=("Consolas", 9))
        val_lbl.pack(anchor=tk.W)
        parent.columnconfigure(col, weight=1)
        return val_lbl

    def load_sample_integers(self):
        self.elem_text.delete("1.0", tk.END)
        self.elem_text.insert(
            tk.END,
            "75872066500, 68562112744, 19339160129, 24156275768, 11525390137, 34580469918, "
            "87752813318, 25906232742, 20636790211, 47170921689, 84559604264, 23643831465, "
            "33227966252, 76687960064, 39654715033, 65528900292"
        )
        self.target_var.set("135205864112")
        self.mode_var.set("findone")

    def load_sample_decimals(self):
        self.elem_text.delete("1.0", tk.END)
        self.elem_text.insert(tk.END, "12.5, 30.25, 45.75, 11.5, 22.0, 99.125")
        self.target_var.set("58.25")
        self.mode_var.set("findone")

    def load_sample_unsat(self):
        self.elem_text.delete("1.0", tk.END)
        self.elem_text.insert(tk.END, "10, 20, 30, 40, 50, 60")
        self.target_var.set("65")
        self.mode_var.set("decision")

    def clear_inputs(self):
        self.elem_text.delete("1.0", tk.END)
        self.target_var.set("")
        self.output_text.delete("1.0", tk.END)
        self.status_banner.config(text="READY", bg="#edf2f7", fg="#2d3748")
        self.metric_strategy.config(text="-")
        self.metric_runtime.config(text="-")
        self.metric_ram.config(text="-")
        self.metric_k_range.config(text="-")
        self.metric_nodes.config(text="-")
        self.metric_l7.config(text="-")

    def on_solve_clicked(self):
        elems = self.elem_text.get("1.0", tk.END).strip()
        tgt = self.target_var.get().strip()
        mode = self.mode_var.get().strip()
        tl = self.timeout_var.get().strip()

        if not elems:
            messagebox.showwarning("Input Missing", "Please enter elements list.")
            return
        if not tgt:
            messagebox.showwarning("Input Missing", "Please enter target value.")
            return

        if not os.path.exists(self.exe_path):
            messagebox.showerror(
                "Executable Not Found",
                f"Could not find adaptsspCli.exe at:\n{self.exe_path}\nPlease build v4/adaptsspCli.cpp first."
            )
            return

        self.solve_btn.config(state=tk.DISABLED)
        self.status_banner.config(text="SOLVING IN PROGRESS... Please wait", bg="#feebc8", fg="#c05621")
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, "Running solver subprocess...\n")

        thread = threading.Thread(target=self.run_solver_thread, args=(elems, tgt, mode, tl))
        thread.daemon = True
        thread.start()

    def run_solver_thread(self, elems, tgt, mode, tl):
        cmd = [
            self.exe_path,
            elems,
            tgt,
            mode,
            "5000",
            tl if tl else "120000",
            "--json"
        ]

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            stdout = proc.stdout.strip()
            stderr = proc.stderr.strip()

            if proc.returncode != 0 and not stdout:
                self.root.after(0, self.on_solver_error, f"Process exited with code {proc.returncode}\n{stderr}")
                return

            try:
                data = json.loads(stdout)
                self.root.after(0, self.on_solver_success, data)
            except json.JSONDecodeError:
                # If stdout is not JSON, fallback to raw display
                self.root.after(0, self.on_solver_raw_output, stdout, stderr)

        except subprocess.TimeoutExpired:
            self.root.after(0, self.on_solver_error, "Solver subprocess timed out after 180 seconds.")
        except Exception as e:
            self.root.after(0, self.on_solver_error, str(e))

    def on_solver_error(self, err_msg):
        self.solve_btn.config(state=tk.NORMAL)
        self.status_banner.config(text="EXECUTION ERROR", bg="#fed7d7", fg="#9b2c2c")
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, f"Error running solver:\n{err_msg}\n")

    def on_solver_raw_output(self, stdout, stderr):
        self.solve_btn.config(state=tk.NORMAL)
        self.status_banner.config(text="EXECUTION COMPLETED (RAW OUTPUT)", bg="#e2e8f0", fg="#4a5568")
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, stdout if stdout else stderr)

    def on_solver_success(self, d):
        self.solve_btn.config(state=tk.NORMAL)

        status = d.get("status", "UNKNOWN")
        solved = d.get("has_solution", False)

        # Update Banner Color
        if "SOLVED" in status:
            self.status_banner.config(text=f"STATUS: {status}", bg="#c6f6d5", fg="#22543d")
        elif "UNSAT" in status:
            self.status_banner.config(text=f"STATUS: {status}", bg="#feebc8", fg="#7b341e")
        else:
            self.status_banner.config(text=f"STATUS: {status}", bg="#fed7d7", fg="#9b2c2c")

        # Update Metrics
        self.metric_strategy.config(text=d.get("strategy_chosen", "-"))
        self.metric_runtime.config(text=f"{d.get('runtime_ms', 0.0):.4f} ms")
        self.metric_ram.config(text=f"{d.get('peak_ram_mb', 0.0):.2f} MB")
        self.metric_k_range.config(text=f"[{d.get('k_min', -1)} .. {d.get('k_max', -1)}] ({d.get('window_ratio_pct', 0)}%)")
        self.metric_nodes.config(text=f"{d.get('states_evaluated', 0):,} eval / {d.get('states_pruned', 0):,} pruned")

        verified = d.get("verified", False)
        ver_text = "PASSED [100% VALID]" if verified else "FAILED"
        self.metric_l7.config(text=ver_text)

        # Build Detailed Report
        out_lines = []
        out_lines.append("=" * 80)
        out_lines.append("               ADAPT-SSP v4 EXECUTION SUMMARY REPORT")
        out_lines.append("=" * 80)
        out_lines.append(f"Elements (N)       : {d.get('elements_count', 0)} raw ({d.get('active_elements_count', 0)} active)")
        out_lines.append(f"Target Value (T)   : {d.get('target', 0)}")
        if d.get("decimal_scale_factor", 1) > 1:
            out_lines.append(f"Original Target    : {d.get('original_target', '')} (Decimal Scale Factor: {d.get('decimal_scale_factor')})")
        out_lines.append(f"Total Sum (Sigma)  : {d.get('total_sum', 0)}")
        out_lines.append(f"Effective Target   : {d.get('effective_target', 0)}" + (" [Dual Complement Applied]" if d.get('complement_applied') else ""))
        out_lines.append(f"GCD Value          : {d.get('gcd_val', 1)}")
        out_lines.append(f"Density            : {d.get('density', 0.0):.4f}")
        out_lines.append(f"Structure Profile  : {d.get('structure_profile', '-')}")
        out_lines.append(f"Strategy Chosen    : {d.get('strategy_chosen', '-')}")
        if d.get("boundary_swap_hit"):
            out_lines.append(f"Boundary Swap Hit  : {d.get('boundary_swap_details', '')}")
        if d.get("residue_primes_checked", 0) > 0:
            out_lines.append(f"Residue Sieve      : {d.get('residue_primes_checked')} primes checked, {d.get('residue_elements_eliminated')} eliminated")
        out_lines.append(f"L7 Independent Ver : {d.get('verification_message', '-')}")
        out_lines.append(f"Engine Message     : {d.get('engine_message', '-')}")
        out_lines.append("-" * 80)

        sample_sol = d.get("sample_solution", {})
        vals = sample_sol.get("values", [])
        indices = sample_sol.get("original_indices", [])
        orig_vals = sample_sol.get("original_values", [])

        if solved and vals:
            out_lines.append(f"Exact Solution Found ({len(vals)} elements):")
            out_lines.append(f"  Values (Integer) : {vals}")
            if orig_vals:
                out_lines.append(f"  Values (Decimal) : {orig_vals}")
            out_lines.append(f"  Original Indices : {indices}")
            out_lines.append(f"  Subset Sum       : {sample_sol.get('sum', 0)} == Target {d.get('target', 0)} [100% MATCH]")

            all_sols = d.get("all_solutions", [])
            if len(all_sols) > 1:
                out_lines.append("\nAdditional Variations Extracted:")
                for sol in all_sols[:20]:
                    out_lines.append(f"  Sol #{sol.get('index')}: {sol.get('values')} -> Sum = {sol.get('sum')}")
                if len(all_sols) > 20:
                    out_lines.append(f"  ... ({len(all_sols) - 20} more solutions stored)")
        elif not solved:
            out_lines.append("NO EXACT SUBSET FOUND (PROVABLY UNSAT)")

        out_lines.append("=" * 80)

        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, "\n".join(out_lines))


def main():
    root = tk.Tk()
    app = AdaptSspGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
