"""Visible local recorder. Microphone starts only when the user clicks Record."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:
    os.environ[key]='1'
from pathlib import Path
import argparse
import json
import queue
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import sounddevice as sd
from .session import ROOT, create_session, read_session, save_clip, evaluate_session


class Recorder:
    def __init__(self, window, folder=None):
        self.window, self.folder = window, Path(folder) if folder else None
        self.draft, self.rate, self.busy = None, None, False
        self.events = queue.Queue()
        window.title('ML-CEP - Meri voice evaluation (20 recordings)')
        window.geometry('850x690')
        window.minsize(760,640)
        style=ttk.Style(); style.theme_use('clam')
        style.configure('TLabel',font=('Segoe UI',11)); style.configure('TButton',font=('Segoe UI',11),padding=9)
        main=ttk.Frame(window,padding=24); main.pack(fill='both',expand=True)
        ttk.Label(main,text='Aapki voice: 20-recording evaluation',font=('Segoe UI',20,'bold')).pack(anchor='w')
        ttk.Label(main,text='5 happy  |  5 disgust  |  5 sad  |  5 angry\nAudio sirf is computer par save hogi. Models in recordings par train nahi honge.',wraplength=780).pack(anchor='w',pady=(8,16))
        devices=sd.query_devices()
        self.inputs=[(i,d) for i,d in enumerate(devices) if d['max_input_channels']>0]
        self.device=ttk.Combobox(main,state='readonly',values=[f'{i}: {d["name"]}' for i,d in self.inputs],width=85)
        self.device.pack(fill='x')
        default=sd.default.device[0]
        if self.inputs:
            self.device.current(next((j for j,(i,_) in enumerate(self.inputs) if i==default),0))
        self.status=tk.StringVar(value='Ready. Session start karein; microphone abhi band hai.')
        self.progress=tk.StringVar()
        self.instruction=tk.StringVar()
        ttk.Label(main,textvariable=self.progress,font=('Segoe UI',14,'bold')).pack(anchor='w',pady=(20,8))
        ttk.Label(main,textvariable=self.instruction,font=('Segoe UI',16),wraplength=780,foreground='#125c70').pack(anchor='w',pady=8)
        ttk.Label(main,text='Har baar yahi sentence bolein: "The airplane is almost full."\nDikhaye gaye emotion mein naturally bolein. Record dabane par 5 seconds milenge.\nEmotion ka naam bolne ki zaroorat nahi. Quiet jagah aur same mic distance rakhein.',wraplength=780).pack(anchor='w',pady=12)
        controls=ttk.Frame(main); controls.pack(fill='x',pady=10)
        self.start=ttk.Button(controls,text='Start session',command=self.begin); self.start.pack(side='left',padx=(0,8))
        self.record=ttk.Button(controls,text='Record / retake (5 sec)',command=self.capture); self.record.pack(side='left',padx=8)
        self.listen=ttk.Button(controls,text='Listen',command=self.play); self.listen.pack(side='left',padx=8)
        self.save=ttk.Button(controls,text='Save & next',command=self.save); self.save.pack(side='left',padx=8)
        self.evaluate=ttk.Button(main,text='Evaluate all 20 recordings',command=self.evaluate_all); self.evaluate.pack(anchor='w',pady=8)
        ttk.Label(main,textvariable=self.status,wraplength=780,foreground='#215767').pack(anchor='w',pady=12)
        self.results=tk.Text(main,height=7,wrap='word',font=('Segoe UI',11),state='disabled',background='#eef5f7',relief='flat')
        self.results.pack(fill='both',expand=True)
        window.protocol('WM_DELETE_WINDOW',self.close)
        self.refresh(); window.after(100,self.poll)

    def begin(self):
        if not self.inputs:
            messagebox.showerror('Microphone','Koi input microphone available nahi. Windows microphone permissions check karein.'); return
        self.folder=create_session(); self.status.set(f'Session: {self.folder.name}'); self.refresh()

    def refresh(self):
        data=read_session(self.folder) if self.folder else None
        count=len(data['clips']) if data else 0
        self.progress.set(f'Saved: {count}/20')
        self.instruction.set('Start session dabayein.' if not data else ('Tamam 20 recordings save ho gayi hain.' if count==20 else f'Agli recording: {count+1}/20  |  EMOTION: '+data['plan'][count].upper().replace('ANGER','ANGRY')))
        active=data is not None and count<20 and not self.busy
        self.start.config(state='normal' if data is None and not self.busy else 'disabled')
        self.record.config(state='normal' if active else 'disabled')
        self.listen.config(state='normal' if self.draft is not None and not self.busy else 'disabled')
        self.save.config(state='normal' if self.draft is not None and active else 'disabled')
        self.evaluate.config(state='normal' if data and count==20 and not self.busy else 'disabled')
        self.device.config(state='readonly' if not self.busy else 'disabled')

    def worker(self, action, function):
        self.busy=True; self.refresh()
        def work():
            try: self.events.put((action,function()))
            except Exception as exc: self.events.put(('error',str(exc)))
        threading.Thread(target=work,daemon=True).start()

    def capture(self):
        i,device=self.inputs[self.device.current()]
        self.rate=int(device['default_samplerate'])
        self.used_device=f'{i}: {device["name"]}'
        self.status.set('RECORDING - ab bolein (5 seconds)...')
        def record():
            audio=sd.rec(int(self.rate*5),samplerate=self.rate,channels=1,dtype='float32',device=i)
            overflow=sd.wait()
            if overflow: raise RuntimeError(f'Audio stream issue: {overflow}. Retake karein.')
            if not np.isfinite(audio).all() or np.max(abs(audio))<1e-4:
                raise ValueError('Silence mili. Mic select aur Windows input permission check karein, phir retake karein.')
            return audio.copy()
        self.worker('recorded',record)

    def play(self):
        self.status.set('Draft playback chal rahi hai...')
        self.worker('played',lambda: (sd.play(self.draft,self.rate),sd.wait()))

    def save(self):
        data=save_clip(self.folder,self.draft,self.rate,self.used_device)
        self.draft=None
        self.status.set(f"Saved {len(data['clips'])}/20. Agle emotion ko dekh kar record karein.")
        self.refresh()

    def evaluate_all(self):
        if not (ROOT/'experiments/representation_v2/selection.json').exists():
            messagebox.showinfo('Training','Improved model ki training abhi chal rahi hai. Recordings mehfooz hain; training complete hone par Evaluate dabayein.'); return
        self.status.set('Dono frozen models evaluate ho rahe hain. Recordings par training nahi ho rahi...')
        self.worker('evaluated',lambda:evaluate_session(self.folder))

    def poll(self):
        try:
            action,value=self.events.get_nowait(); self.busy=False
            if action=='error':
                self.status.set('Issue: '+value); messagebox.showerror('Audio / evaluation issue',value)
            elif action=='recorded':
                self.draft=value
                clipping=float(np.mean(abs(value)>=.999))
                self.status.set('Draft tayyar. Listen karein; theek ho to Save & next, warna retake.'+(' Clipping mili; mic gain kam karke retake behtar hai.' if clipping>.001 else ''))
            elif action=='played': self.status.set('Playback complete. Save & next ya retake karein.')
            elif action=='evaluated':
                lines=['20 REAL recordings - personal voice results']
                for model,metrics in value['models'].items():
                    lines.append(f"{model}: {metrics['correct']}/20 correct | accuracy {metrics['accuracy']:.1%} | 4-class macro-F1 {metrics['macro_f1_four_supported']:.3f}")
                lines+=['Ek speaker ka personal test; general performance guarantee nahi.',f'Results: {self.folder}']
                self.results.config(state='normal'); self.results.delete('1.0','end'); self.results.insert('1.0','\n'.join(lines)); self.results.config(state='disabled')
                self.status.set('Evaluation complete. WAVs, predictions.csv, evaluation.json aur report.md locally save ho gaye.')
            self.refresh()
        except queue.Empty: pass
        self.window.after(100,self.poll)

    def close(self):
        if self.busy and not messagebox.askyesno('Close','Recording/playback/evaluation chal rahi hai. Window band karni hai?'): return
        sd.stop(); self.window.destroy()


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--resume',type=Path); parser.add_argument('--check',action='store_true'); args=parser.parse_args()
    if args.check:
        print(sd.query_devices())
    else:
        window=tk.Tk(); Recorder(window,args.resume); window.mainloop()
