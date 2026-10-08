//! Scoped progress events for the thread that owns a model or compiler.
use std::{cell::RefCell, rc::Rc};

type Listener = Rc<dyn Fn(&str)>;
thread_local! {
    static LISTENER: RefCell<Option<Listener>> = const { RefCell::new(None) };
}

// Rc keeps this guard on the thread whose listener it restores.
pub(crate) struct Scope(Option<Listener>);

pub(crate) fn listen(listener: impl Fn(&str) + 'static) -> Scope {
    Scope(LISTENER.with(|slot| slot.replace(Some(Rc::new(listener)))))
}

impl Drop for Scope {
    fn drop(&mut self) {
        LISTENER.with(|slot| slot.replace(self.0.take()));
    }
}

/// Returns whether a listener received the event. Ordinary CLI output is unchanged.
pub(crate) fn emit(message: impl AsRef<str>) -> bool {
    let listener = LISTENER.with(|slot| slot.borrow().clone());
    if let Some(listener) = listener {
        listener(message.as_ref());
        true
    } else {
        false
    }
}

pub(crate) fn warning(message: impl AsRef<str>) {
    if !emit(message.as_ref()) {
        eprintln!("{}", message.as_ref());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn listeners_are_thread_local_and_nested_scopes_restore_the_previous_listener() {
        assert!(!emit("outside"));
        let events = Rc::new(RefCell::new(Vec::new()));
        let sink = events.clone();
        let outer = listen(move |s| sink.borrow_mut().push(s.to_owned()));
        assert!(emit("outer"));
        assert!(!std::thread::spawn(|| emit("other thread")).join().unwrap());
        {
            let sink = events.clone();
            let _inner = listen(move |s| sink.borrow_mut().push(format!("inner: {s}")));
            emit("nested");
        }
        emit("restored");
        drop(outer);
        assert!(!emit("outside"));
        assert_eq!(*events.borrow(), ["outer", "inner: nested", "restored"]);
    }
}
