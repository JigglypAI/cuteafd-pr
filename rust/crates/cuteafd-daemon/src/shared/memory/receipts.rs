//! Passive owner inventory. Every extent and address comes from retained storage.
use anyhow::{ensure, Context, Result};
use cuteafd_core::memory_layout::Category;
use cuteafd_ffi::{CuteafdDeviceBuffer, CuteafdHostBuffer};
use cuteafd_loader::placement::planner::{Allocation, CoordinatorDevices, Lifetime, Location, Receipt, View};

pub(crate) struct Inventory<'a> {
    pub receipts: Vec<Receipt>,
    devices: CoordinatorDevices<'a>,
    prefix: String,
    lifetime: Lifetime,
    category: Category,
    group: String,
}
impl<'a> Inventory<'a> {
    pub(crate) fn new(devices: CoordinatorDevices<'a>, prefix: &str, lifetime: Lifetime,
        category: Category, group: &str) -> Self {
        Self { receipts: Vec::new(), devices, prefix: prefix.into(), lifetime, category, group: group.into() }
    }
    pub(crate) fn location(&self, physical_device: i32) -> Result<Location> {
        Ok(self.devices.location(physical_device)?)
    }
    pub(crate) fn lifetime(&self) -> Lifetime { self.lifetime }
    pub(crate) fn extend(&mut self, receipts: Vec<Receipt>) -> Result<()> {
        for mut receipt in receipts {
            receipt.allocation.id = format!("{}.{}", self.prefix, receipt.allocation.id);
            if let Some(view) = &mut receipt.allocation.view {
                view.backing = format!("{}.{}", self.prefix, view.backing);
            }
            ensure!(!self.receipts.iter().any(|r| r.allocation.id == receipt.allocation.id),
                "duplicate imported owner {}", receipt.allocation.id);
            self.receipts.push(receipt);
        }
        Ok(())
    }
    fn add(&mut self, name: &str, location: Location, address: u64, bytes: u64) -> Result<()> {
        ensure!(!name.is_empty() && address != 0 && bytes != 0, "empty actual allocation {name}");
        let id = format!("{}.{name}", self.prefix);
        ensure!(!self.receipts.iter().any(|r| r.allocation.id == id), "duplicate owner {id}");
        self.receipts.push(Receipt { allocation: Allocation::buffer(id, location, bytes,
            self.lifetime, self.category, &self.group), address });
        Ok(())
    }
    pub(crate) fn device(&mut self, name: &str, buffer: CuteafdDeviceBuffer) -> Result<()> {
        self.add(name, self.devices.location(buffer.device_id)?, buffer.ptr as u64, buffer.bytes as u64)
    }
    pub(crate) fn backed_pinned(&mut self, name: &str, buffer: CuteafdHostBuffer,
        owner: CuteafdHostBuffer) -> Result<()> {
        let offset = (buffer.ptr as u64).checked_sub(owner.ptr as u64).context("host view before backing")?;
        ensure!(buffer.bytes > 0 && offset.checked_add(buffer.bytes as u64).is_some_and(|n| n <= owner.bytes as u64),
            "host view {name} exceeds backing");
        let backing = self.receipts.iter().find(|receipt| receipt.allocation.view.is_none()
            && receipt.allocation.location == Location::PINNED && receipt.address == owner.ptr as u64
            && receipt.allocation.bytes == owner.bytes as u64).map(|r| r.allocation.id.clone());
        let backing = match backing {
            Some(id) => id,
            None => {
                self.pinned(&format!("{name}.backing"), owner)?;
                format!("{}.{name}.backing", self.prefix)
            }
        };
        self.pinned(name, buffer)?;
        self.receipts.last_mut().unwrap().allocation.view = Some(View { backing, offset });
        Ok(())
    }
    pub(crate) fn pinned(&mut self, name: &str, buffer: CuteafdHostBuffer) -> Result<()> {
        self.add(name, Location::PINNED, buffer.ptr as u64, buffer.bytes as u64)
    }
    pub(crate) fn host_backing(&mut self, name: &str, address: u64, bytes: u64) -> Result<()> {
        if bytes != 0 { self.add(name, Location::HOST, address, bytes)?; }
        Ok(())
    }
    pub(crate) fn host<T>(&mut self, name: &str, values: &Vec<T>) -> Result<()> {
        let bytes = values.capacity().checked_mul(std::mem::size_of::<T>()).context("host capacity overflow")?;
        if bytes != 0 { self.add(name, Location::HOST, values.as_ptr() as u64, bytes as u64)?; }
        Ok(())
    }
    pub(crate) fn view(&mut self, name: &str, buffer: CuteafdDeviceBuffer, backing: &str,
        owner: CuteafdDeviceBuffer) -> Result<()> {
        ensure!(buffer.device_id == owner.device_id, "view {name} changed device");
        let offset = (buffer.ptr as u64).checked_sub(owner.ptr as u64).context("view before owner")?;
        ensure!(offset.checked_add(buffer.bytes as u64).is_some_and(|end| end <= owner.bytes as u64),
            "view {name} exceeds actual backing");
        self.device(name, buffer)?;
        self.receipts.last_mut().unwrap().allocation.view = Some(View {
            backing: format!("{}.{backing}", self.prefix), offset,
        });
        Ok(())
    }
    pub(crate) fn backed_device(&mut self, name: &str, buffer: CuteafdDeviceBuffer,
        owner: CuteafdDeviceBuffer) -> Result<()> {
        let location = self.devices.location(owner.device_id)?;
        let backing = self.receipts.iter().find(|receipt| receipt.allocation.view.is_none()
            && receipt.allocation.location == location && receipt.address == owner.ptr as u64
            && receipt.allocation.bytes == owner.bytes as u64).map(|r| r.allocation.id.clone());
        let backing = match backing {
            Some(id) => id,
            None => {
                let id = format!("{name}.backing");
                self.device(&id, owner)?;
                format!("{}.{id}", self.prefix)
            }
        };
        let offset = (buffer.ptr as u64).checked_sub(owner.ptr as u64).context("region before owner")?;
        ensure!(buffer.device_id == owner.device_id
            && offset.checked_add(buffer.bytes as u64).is_some_and(|end| end <= owner.bytes as u64),
            "region {name} outside actual owner");
        self.device(name, buffer)?;
        self.receipts.last_mut().unwrap().allocation.view = Some(View { backing, offset });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn views_derive_real_offsets_and_host_capacity_is_not_length() -> Result<()> {
        let devices = [1, 0];
        let mut inventory = Inventory::new(CoordinatorDevices::new(&devices)?, "fixture",
            Lifetime::TARGET, Category::Workspace, "fixture");
        let owner = CuteafdDeviceBuffer { ptr: std::ptr::without_provenance_mut(0x1000), bytes: 1024, device_id: 0, flags: 0 };
        inventory.device("arena", owner)?;
        inventory.view("slice", CuteafdDeviceBuffer { ptr: std::ptr::without_provenance_mut(0x1100), bytes: 128,
            ..owner }, "arena", owner)?;
        assert_eq!(inventory.receipts[1].allocation.view, Some(View { backing: "fixture.arena".into(), offset: 256 }));
        assert!(inventory.view("past", CuteafdDeviceBuffer { bytes: 2048, ..owner }, "arena", owner).is_err());
        let mut host = Vec::<u32>::with_capacity(11);
        host.push(3);
        inventory.host("host", &host)?;
        assert_eq!(inventory.receipts[2].allocation.bytes, (host.capacity() * 4) as u64);
        Ok(())
    }
}
