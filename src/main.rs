use candle_datasets::hub::from_hub as load_dataset;
use hf_hub::api::sync::ApiBuilder; // 変更
use parquet::file::reader::FileReader;
use parquet::record::RowAccessor;
use rand;
use std::path::PathBuf; // 追加。パスの指定用
use tqdm::tqdm;

fn ascii_printable() -> Vec<char> {
    (32..127).chain((10..11)).map(|i| i as u8 as char).collect()
}

fn ascii_to_id(ch: char) -> usize {
    if ch == '\n' || ch == '\r' {
        95
    } else if ch.is_ascii() && ch as u32 >= 32 {
        ch as usize - 32
    } else {
        96
    }
}

fn eval_linear_bounded_automaton(
    s: &str,
    table: Vec<Vec<Vec<(usize, usize, usize, usize)>>>,
) -> Vec<usize> {
    let mut state = 0;
    let mut output_history: Vec<usize> = Vec::new();
    let mut tape = vec![0; s.len()];
    let mut location = 0;
    let mut c = 0;
    let mut d = 0;
    while c < s.len() {
        let ch = ascii_to_id(s.chars().nth(c).unwrap());
        let (next_state, next_tape, next_location, output) = table[state][ch][tape[location]];
        state = next_state;
        tape[location] = next_tape;
        if next_location == 1 {
            if (location == tape.len() - 1) {
                tape.push(0);
            }
            location += 1;
        } else if next_location == 2 {
            if (location != 0) {
                location -= 1;
            } else {
                tape.insert(0, 0);
            }
        }
        if (output != 2) {
            output_history.push(output);
            c += 1;
            d = 0;
        } else {
            d += 1;
            if (d > s.len() * 32) {
                output_history.push(2);
                c += 1;
                d = 0;
            }
        }
    }
    output_history
}

fn initialize_random_table(num_states: usize) -> Vec<Vec<Vec<(usize, usize, usize, usize)>>> {
    let mut table = vec![vec![vec![(0, 0, 0, 0); num_states]; 96]; num_states];
    for i in 0..num_states {
        for j in 0..96 {
            for k in 0..num_states {
                table[i][j][k] = (
                    rand::Rng::gen_range(&mut rand::thread_rng(), 0..num_states),
                    rand::Rng::gen_range(&mut rand::thread_rng(), 0..num_states),
                    rand::Rng::gen_range(&mut rand::thread_rng(), 0..3),
                    rand::Rng::gen_range(&mut rand::thread_rng(), 0..3),
                );
            }
        }
    }
    table
}

fn main() {
    let builder = ApiBuilder::new();
    let builder = builder.with_cache_dir(PathBuf::from("./cache"));

    //let api = builder.build().unwrap();
    //let repo_name = "ronantakizawa/github-top-code".to_string();
    //let ds = load_dataset(&api, repo_name).unwrap();

    let table = initialize_random_table(8);
    println!(
        "{:?}",
        eval_linear_bounded_automaton("hello\nworld", table.clone())
    );
    println!(
        "{:?}",
        eval_linear_bounded_automaton(
            "abeshinzo \n is hitler abeshinzo is hitler abeshinzo is hitler",
            table.clone()
        )
    );

    /*while let Some(file) = ds.iter().next() {
        let schema = file.metadata().file_metadata().schema();
        if let Ok(row_iter) = file.get_row_iter(Some(schema.clone())) {
            let input_index = schema
                .get_fields()
                .iter()
                .position(|f| f.name() == "content")
                .expect("column 'content' not found");
            for row in tqdm(row_iter) {
                if let Ok(row) = row {
                    println!("{}", row.get_string(input_index).unwrap().to_string());
                }
            }
        }
    }*/
}
