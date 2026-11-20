'use strict';
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),path=require('node:path');
const code=fs.readFileSync(path.join(__dirname,'../web/desktop_webview.js'),'utf8');
const html=fs.readFileSync(path.join(__dirname,'../web/desktop_webview.html'),'utf8');
assert.match(html,/id="modelSort"/);
assert.match(html,/<option value="popular" selected>/);
const start=code.indexOf('const roomMetric=');
const end=code.indexOf('function statusClass',start);
assert(start>0&&end>start);
let mode='popular',scope='public';
const models=[
  {room_id:'a',model:'A',datasets:[],on_wan:true,online:6,cumulative_participants:3,training_seconds:60*60,created_at:100},
  {room_id:'b',model:'B',datasets:[],on_wan:true,online:2,cumulative_participants:600,training_seconds:3600*100,created_at:200},
  {room_id:'c',model:'C',datasets:['1','2'],on_wan:true,online:3,cumulative_participants:40,training_seconds:3600*4,created_at:300},
  {room_id:'d',model:'D',datasets:[],on_wan:true,online:0,created_at:0},
];
const context=vm.createContext({models,get:which=>which==='modelSort'?mode:which==='modelFilter'?'':which==='modelScope'?scope:'internet',Math,Number,String});
vm.runInContext(code.slice(start,end),context);
const ids=()=>Array.from(vm.runInContext('modelFiltered().map(x=>x.room_id)',context));
assert.equal(ids()[0],'b','popular should use real history, not merely current online count');
mode='online';assert.equal(ids()[0],'a');
mode='participants';assert.equal(ids()[0],'b');
mode='training';assert.equal(ids()[0],'b');
mode='newest';assert.equal(ids()[0],'c');
mode='oldest';assert.deepEqual(ids().slice(0,2),['a','b']);
mode='datasets';assert.equal(ids()[0],'c');
mode='name';assert.deepEqual(ids(),['a','b','c','d']);
// Stale/missing metrics must not cause NaN comparator instability.
mode='popular';assert.equal(ids()[3],'d');
console.log('PASS v143 eight stable sorts, popularity, missing legacy metrics');
